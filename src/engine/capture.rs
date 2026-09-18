//! Rendering a world into memory instead of into a window.
//!
//! This exists so that "does it actually look right" stops being a question
//! only a person at the screen can answer. It renders with the same [`Scene`]
//! the window uses, reads the pixels back, and hands over an [`Image`] that a
//! test can make assertions about - or that can be written out as a PNG and
//! looked at.

use colby_core::{Err, Result, abi::World, err, glam::Vec3};
use wgpu::{
	BufferDescriptor, BufferUsages, Extent3d, MapMode, PollType, TexelCopyBufferInfo,
	TexelCopyBufferLayout, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
	TextureViewDescriptor,
};

use crate::{gpu::Gpu, image::Image, overlay::Overlay, scene::Scene};

/// The color format a capture renders into when nobody says otherwise.
///
/// The same family the window uses, so that the numbers a test reads back are
/// the numbers that would reach the screen. A capture drawn beside a window
/// takes the window's own format instead, @ref [`Capture::in_format`].
const CAPTURE_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// Copying a texture to a buffer wants each row aligned to this.
const ROW_ALIGNMENT: u32 = 256;

/// An offscreen target and everything needed to draw into it.
pub struct Capture {
	scene: Scene,
	width: u32,
	height: u32,
	/// What the target is, and therefore how the bytes read back are laid out.
	format: TextureFormat,
	color: wgpu::Texture,
	readback: wgpu::Buffer,
	/// Bytes per row in the readback buffer, padded up to [`ROW_ALIGNMENT`].
	padded_stride: u32,
}

impl Capture {
	/// Builds an offscreen renderer on the shared device.
	///
	/// A scene of its own on the same device: what it uploads it uploads for
	/// itself, on demand, and what it draws can be read by anything else on
	/// that device without a trip through the CPU.
	///
	/// @param gpu - the device to draw with
	/// @param width - the image width in pixels
	/// @param height - the image height in pixels
	pub fn new(gpu: &Gpu, width: u32, height: u32) -> Result<Self> {
		Self::in_format(gpu, CAPTURE_FORMAT, width, height)
	}

	/// The same, into a format somebody else chose.
	///
	/// For a picture taken beside a window: an overlay's pipeline is built for
	/// the window's format, and it cannot draw into a target of another. Only
	/// the eight-bit formats are taken, because the readback has to know what
	/// a texel is; a blue-first one is turned round into the red-first order
	/// an [`Image`] holds on the way out.
	///
	/// @param gpu - the device to draw with
	/// @param format - the color format the target has
	/// @param width - the image width in pixels
	/// @param height - the image height in pixels
	pub fn in_format(gpu: &Gpu, format: TextureFormat, width: u32, height: u32) -> Result<Self> {
		if !readable(format) {
			return Err!(Graphics("a capture reads back eight-bit RGBA or BGRA, not {format:?}"));
		}

		let (width, height) = (width.max(1), height.max(1));
		let device = gpu.device();

		let color = device.create_texture(&TextureDescriptor {
			label: Some("capture"),
			size: Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format,
			usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
			view_formats: &[],
		});

		let padded_stride = width.saturating_mul(4).div_ceil(ROW_ALIGNMENT) * ROW_ALIGNMENT;
		let readback = device.create_buffer(&BufferDescriptor {
			label: Some("readback"),
			size: u64::from(padded_stride) * u64::from(height),
			usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});

		let scene = Scene::new(gpu, format, width, height)?;

		Ok(Self {
			scene,
			width,
			height,
			format,
			color,
			readback,
			padded_stride,
		})
	}

	/// Renders a world and reads the result back.
	///
	/// @param world - the state to draw; its `aspect` is overwritten to match
	/// the capture, since a mismatched one would stretch the picture
	/// @return the pixels, top row first
	pub fn shoot(&mut self, world: &mut World) -> Result<Image> {
		self.shoot_with(world, &mut [])
	}

	/// The same, with something drawn over the scene.
	///
	/// What makes the game's interface testable by pixels rather than only by
	/// looking at a window: `--shot` draws through this, so a screenshot shows
	/// what the screen shows.
	///
	/// @param world - the state to draw
	/// @param overlays - drawn in order, after the scene
	/// @return the pixels, top row first
	pub fn shoot_with(
		&mut self,
		world: &mut World,
		overlays: &mut [&mut dyn Overlay],
	) -> Result<Image> {
		self.draw(world, overlays);
		self.copy_out();

		self.read_back()
	}

	/// Draws a frame and leaves it on the GPU.
	///
	/// What [`shoot_with`](Self::shoot_with) does without the three and a half
	/// megabytes of readback, which a measuring run must not pay for: a copy
	/// to a mappable buffer is not part of any frame anybody plays, and it
	/// would be by far the largest thing in a table of what a frame costs.
	///
	/// @param world - the state to draw; its `aspect` is overwritten to match
	/// @param overlays - drawn in order, after the scene
	pub fn draw(&mut self, world: &mut World, overlays: &mut [&mut dyn Overlay]) {
		world.aspect = self.aspect();

		let view = self
			.color
			.create_view(&TextureViewDescriptor::default());

		// no time at all, which is exactly right: a capture is a fresh surface
		// with no history, so its eye is set to what it measures rather than
		// moved towards it, and the answer does not depend on a clock.
		self.scene.render(&view, world, None, 0.0);

		for overlay in overlays {
			overlay.draw(self.scene.device(), self.scene.queue(), &view, self.width, self.height);
		}
	}

	/// Draws a frame into one rectangle of the target, the way a window with
	/// tools around its picture draws the world into the middle. A test's.
	///
	/// @param world - the state to draw; its `aspect` is overwritten to match
	/// the rectangle
	/// @param view - the rectangle
	#[cfg(test)]
	pub(crate) fn draw_within(&mut self, world: &mut World, view: crate::Viewport) {
		world.aspect = view.aspect();

		let target = self
			.color
			.create_view(&TextureViewDescriptor::default());

		self.scene.render(&target, world, Some(view), 0.0);
	}

	/// Draws a frame into one rectangle of the target and reads the whole
	/// target back. A test's.
	///
	/// @param world - the state to draw; its `aspect` is overwritten to match
	/// the rectangle
	/// @param view - the rectangle
	/// @return the pixels of the whole target, top row first
	#[cfg(test)]
	pub(crate) fn shoot_within(
		&mut self,
		world: &mut World,
		view: crate::Viewport,
	) -> Result<Image> {
		self.draw_within(world, view);
		self.copy_out();

		self.read_back()
	}

	/// The device this capture draws with.
	///
	/// For an [`Overlay`], which builds its pipelines against the same device
	/// rather than a second one.
	#[must_use]
	pub const fn device(&self) -> &wgpu::Device { self.scene.device() }

	/// The queue its work is submitted on.
	#[must_use]
	pub const fn queue(&self) -> &wgpu::Queue { self.scene.queue() }

	/// The color format an overlay has to build a pipeline for.
	#[must_use]
	pub const fn format(&self) -> TextureFormat { self.format }

	/// What this capture draws with.
	///
	/// Exposed so that a test can put a different shader in front of a known
	/// scene and look at what comes out. @ref
	/// [`Scene::set_shader`](crate::Scene::set_shader).
	pub const fn scene_mut(&mut self) -> &mut Scene { &mut self.scene }

	/// The capture's width divided by its height.
	#[must_use]
	#[expect(
		clippy::as_conversions,
		clippy::cast_precision_loss,
		reason = "captures are a few hundred pixels across, nowhere near where f32 stops \
		          representing integers exactly"
	)]
	fn aspect(&self) -> f32 { self.width.max(1) as f32 / self.height.max(1) as f32 }

	/// Records and submits the texture-to-buffer copy.
	fn copy_out(&self) {
		let mut encoder = self
			.scene
			.device()
			.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });

		encoder.copy_texture_to_buffer(
			self.color.as_image_copy(),
			TexelCopyBufferInfo {
				buffer: &self.readback,
				layout: TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(self.padded_stride),
					rows_per_image: Some(self.height),
				},
			},
			Extent3d {
				width: self.width,
				height: self.height,
				depth_or_array_layers: 1,
			},
		);

		self.scene.queue().submit([encoder.finish()]);
	}

	/// Maps the readback buffer and unpads it into an [`Image`].
	fn read_back(&self) -> Result<Image> {
		let slice = self.readback.slice(..);
		slice.map_async(MapMode::Read, |_| {});

		self.scene
			.device()
			.poll(PollType::Wait { submission_index: None, timeout: None })
			.map_err(|error| err!(Graphics("waiting for the readback: {error}")))?;

		let view = slice
			.get_mapped_range()
			.map_err(|error| err!(Graphics("mapping the readback: {error}")))?;

		let stride = usize::try_from(self.width).unwrap_or(0) * 4;
		let padded = usize::try_from(self.padded_stride).unwrap_or(stride);
		let mut pixels = Vec::with_capacity(stride * usize::try_from(self.height).unwrap_or(0));

		for row in view.chunks(padded) {
			let Some(row) = row.get(..stride) else {
				break;
			};

			pixels.extend_from_slice(row);
		}

		// into the order an image holds, if the target was the other way round.
		if swapped(self.format) {
			for texel in pixels.chunks_exact_mut(4) {
				texel.swap(0, 2);
			}
		}

		drop(view);
		self.readback.unmap();

		Ok(Image {
			width: self.width,
			height: self.height,
			pixels,
		})
	}
}

/// Whether a capture can read a format back: four bytes a texel, one a channel.
const fn readable(format: TextureFormat) -> bool {
	matches!(
		format,
		TextureFormat::Rgba8Unorm
			| TextureFormat::Rgba8UnormSrgb
			| TextureFormat::Bgra8Unorm
			| TextureFormat::Bgra8UnormSrgb
	)
}

/// Whether a format's texels are blue first, and have to be turned round.
const fn swapped(format: TextureFormat) -> bool {
	matches!(format, TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb)
}

/// How far apart two colors are, as the largest difference on any channel.
///
/// @param left - one color
/// @param right - the other
/// @return `0` for identical, `255` for opposite
#[must_use]
pub fn distance(left: [u8; 4], right: [u8; 4]) -> u8 {
	left.iter()
		.zip(right)
		.map(|(one, other)| one.abs_diff(other))
		.max()
		.unwrap_or(0)
}

/// Which channel of a color is the largest.
///
/// @param color - the color to look at
/// @return `0` for red, `1` for green, `2` for blue
#[must_use]
pub fn dominant(color: [u8; 4]) -> usize {
	let channels = [color[0], color[1], color[2]];
	let mut best = 0;
	for (index, value) in channels.iter().enumerate() {
		if *value > channels[best] {
			best = index;
		}
	}

	best
}

/// A color as the renderer will see it, for building test worlds.
#[must_use]
pub const fn rgb(red: f32, green: f32, blue: f32) -> Vec3 { Vec3::new(red, green, blue) }

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			Decal, EntityId, Material, MeshData, MeshId, PaintVertex, Pose, PoseId, Post,
			Renderable, SkinVertex, Sky, SkyKind, Texel, TextureData, TextureId, ToneMap,
			Transform,
			cvar::Value,
			material::{Blend, MaterialId},
			mesh,
			skeleton::{Bone, SkeletonData},
		},
		glam::{Mat4, Quat, Vec2, Vec4},
	};
	use wgpu::Backends;

	use super::*;
	use crate::{
		cull, env,
		scene::MSAA,
		shadow,
		skin::{self, Joints},
	};

	/// How big the test captures are. Small enough to be quick, large enough
	/// that a sample well inside a shape is unambiguous.
	const SIZE: (u32, u32) = (320, 240);

	/// The side of the one capture that has to be square. @ref
	/// [`a_texture_reaches_the_screen_the_way_up_its_coordinates_say`].
	const SQUARE: u32 = 256;

	/// How far above the origin the overhead camera sits.
	const HEIGHT: f32 = 5.0;

	/// What every test world sets: the picture squeezed by nothing and exposed
	/// at one.
	///
	/// **Every test in this file is about shading, not about metering.** A
	/// curve is not linear and a measured exposure depends on the average of
	/// the whole picture, so with either of them on, a pixel compared against
	/// the same pixel in a second capture of a slightly different world is a
	/// comparison of two exposures rather than of two surfaces. Turning both
	/// off is what keeps these tests measuring the thing they are named after;
	/// the post-processing has tests of its own.
	fn plainly(world: &mut World) {
		world.post.tonemap = ToneMap::None;
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
	}

	/// A world with a camera looking at the origin from `+z`, and a light
	/// traveling the same way the camera does.
	///
	/// With no ambient at all, a surface facing the camera is fully lit and one
	/// facing away is black. That is what makes the winding test decisive.
	fn looking_world() -> World {
		let mut world = World::new();
		plainly(&mut world);
		world.clear = rgb(0.0, 0.0, 0.2);
		world.ambient = Vec3::ZERO;
		world.light = Vec3::NEG_Z;
		world.camera.position = Vec3::new(0.0, 0.0, 5.0);
		world.camera.target = Vec3::ZERO;

		world
	}

	/// A capture on the binary's one device, or `None` with no GPU.
	///
	/// The device is not handed back and does not need to be: it outlives
	/// every test on it. @ref [`crate::gpu::shared`] for why there is one.
	fn capture_of(width: u32, height: u32) -> Option<Capture> {
		let gpu = crate::gpu::shared()?;

		match Capture::new(gpu, width, height) {
			| Ok(capture) => Some(capture),
			| Err(error) => panic!("building the capture failed: {error}"),
		}
	}

	/// A capture the usual size, or `None` when this machine has no GPU.
	fn capture() -> Option<Capture> { capture_of(SIZE.0, SIZE.1) }

	#[test]
	fn two_captures_on_one_device_each_draw_their_own_picture() {
		// the whole point of the shared device: a second target on the same
		// device is a second scene and not a second GPU, and neither picture
		// leaks into the other. Different sizes and different clears, so a
		// mix-up of targets, sizes or worlds each shows on its own.
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut wide = Capture::new(gpu, 64, 32).expect("the first capture builds");
		let mut tall = Capture::new(gpu, 32, 64).expect("the second capture builds");

		let mut red = World::new();
		plainly(&mut red);
		red.clear = rgb(1.0, 0.0, 0.0);
		let mut blue = World::new();
		plainly(&mut blue);
		blue.clear = rgb(0.0, 0.0, 1.0);

		let first = wide
			.shoot(&mut red)
			.expect("the wide frame renders");
		let second = tall
			.shoot(&mut blue)
			.expect("the tall frame renders");
		let again = wide
			.shoot(&mut red)
			.expect("the wide frame renders again");

		assert_eq!((first.width, first.height), (64, 32), "the wide one is wide");
		assert_eq!((second.width, second.height), (32, 64), "the tall one is tall");
		assert_eq!(dominant(first.pixel(10, 10)), 0, "the wide one is red");
		assert_eq!(dominant(second.pixel(10, 10)), 2, "the tall one is blue");
		assert_eq!(first.pixels, again.pixels, "drawing the other did not touch it");
	}

	#[test]
	fn a_capture_in_the_windows_format_reads_back_the_same_picture() {
		// the window's surface is blue-first on this machine and a capture's
		// is red-first; a screenshot taken beside the window is drawn into the
		// window's format so that the interface's pipeline fits, and the bytes
		// have to come out in the order an image holds. A red cube on a blue
		// clear: a readback that forgot to turn the texels round swaps the two.
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut usual = Capture::new(gpu, SIZE.0, SIZE.1).expect("the usual capture builds");
		let mut windows = Capture::in_format(gpu, TextureFormat::Bgra8UnormSrgb, SIZE.0, SIZE.1)
			.expect("the window's format builds");

		let mut world = looking_world();
		let cube = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		let first = usual
			.shoot(&mut world)
			.expect("the usual frame renders");
		let second = windows
			.shoot(&mut world)
			.expect("the other frame renders");

		assert_eq!(windows.format(), TextureFormat::Bgra8UnormSrgb, "the format is kept");
		assert_eq!(
			dominant(second.pixel(SIZE.0 / 2, SIZE.1 / 2)),
			0,
			"the cube is red either way"
		);
		assert_eq!(dominant(second.pixel(1, 1)), 2, "and the clear is blue either way");
		assert_eq!(first.pixels, second.pixels, "the same picture, byte for byte");
	}

	#[test]
	fn a_format_the_readback_cannot_read_is_refused() {
		let Some(gpu) = crate::gpu::shared() else {
			return;
		};

		for format in
			[TextureFormat::R8Unorm, TextureFormat::Rgba16Float, TextureFormat::Rgb10a2Unorm]
		{
			assert!(
				Capture::in_format(gpu, format, 4, 4).is_err(),
				"{format:?} is not four bytes of one channel each"
			);
		}
	}

	#[test]
	fn an_empty_world_is_nothing_but_the_clear_color() {
		let Some(mut capture) = capture() else {
			eprintln!("no GPU adapter; skipping the pixel tests");
			return;
		};

		let mut world = looking_world();
		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		let corner = image.pixel(1, 1);
		for (x, y) in [(SIZE.0 / 2, SIZE.1 / 2), (SIZE.0 - 2, 1), (1, SIZE.1 - 2)] {
			assert!(
				distance(image.pixel(x, y), corner) <= 1,
				"nothing was spawned, so ({x}, {y}) should match the corner: {:?} against {:?}",
				image.pixel(x, y),
				corner
			);
		}
	}

	#[test]
	fn a_cube_is_drawn_lit_side_towards_the_camera() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		let cube = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);
		let corner = image.pixel(1, 1);

		assert!(distance(middle, corner) > 20, "the cube covers the middle: {middle:?}");
		assert_eq!(dominant(middle), 0, "and it is the red the game asked for: {middle:?}");

		// @note: this is the culling and winding check. The light travels the
		// way the camera looks, so the face turned towards the camera is fully
		// lit and the inside of the far face is black. If the winding were the
		// other way round, back-face culling would drop the near faces and this
		// would be the dark one.
		assert!(
			middle[0] > 120,
			"the near face is lit, so the front of the cube is what survived culling: {middle:?}"
		);
	}

	#[test]
	fn the_nearer_cube_wins_the_depth_test() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let far = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 0.0, -2.0)));
		world
			.entities
			.set_renderable(far, Renderable::new(MeshId::CUBE, rgb(0.0, 0.0, 0.9)));

		let near = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 0.0, 2.0)));
		world
			.entities
			.set_renderable(near, Renderable::new(MeshId::CUBE, rgb(0.0, 0.9, 0.0)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(
			dominant(middle),
			1,
			"the green cube is nearer, so it is the one on screen: {middle:?}"
		);
	}

	#[test]
	fn the_picture_is_neither_mirrored_nor_upside_down() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let right = world
			.entities
			.spawn_at(Transform::at(Vec3::new(1.5, 0.0, 0.0)));
		world
			.entities
			.set_renderable(right, Renderable::new(MeshId::CUBE, rgb(0.9, 0.0, 0.0)));

		let above = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 1.5, 0.0)));
		world
			.entities
			.set_renderable(above, Renderable::new(MeshId::CUBE, rgb(0.0, 0.9, 0.0)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		// a cube at +x belongs on the right of the image.
		let east = image.pixel(SIZE.0 * 3 / 4, SIZE.1 / 2);
		let west = image.pixel(SIZE.0 / 4, SIZE.1 / 2);

		assert_eq!(dominant(east), 0, "the +x cube is on the right: {east:?}");
		assert!(distance(west, image.pixel(1, 1)) <= 1, "and nothing is on the left: {west:?}");

		// a cube at +y belongs at the top, where the row index is small. Get
		// the projection's y convention backwards and this is the assertion
		// that says so.
		let north = image.pixel(SIZE.0 / 2, SIZE.1 / 4);
		let south = image.pixel(SIZE.0 / 2, SIZE.1 * 3 / 4);

		assert_eq!(dominant(north), 1, "the +y cube is at the top: {north:?}");
		assert!(
			distance(south, image.pixel(1, 1)) <= 1,
			"and nothing is at the bottom: {south:?}"
		);
	}

	/// The middle of everything in a row that is not the background.
	///
	/// Cheaper to reason about than a projection worked out by hand, and it
	/// does not care what the field of view is: whatever is drawn, this says
	/// where it is.
	///
	/// @param image - the captured frame
	/// @param row - which scanline to look along
	/// @param background - the color to treat as nothing
	/// @return the middle column of the run of foreground, if there was any
	fn shape_center(image: &Image, row: u32, background: [u8; 4]) -> Option<u32> {
		let (mut first, mut last) = (None, 0);

		for x in 0..image.width {
			if distance(image.pixel(x, row), background) > 20 {
				first.get_or_insert(x);
				last = x;
			}
		}

		first.map(|first| first.midpoint(last))
	}

	/// The highest row of the picture that has anything but the background in
	/// it, counting down from the top.
	///
	/// The same idea [`shape_center`] is and for the same reason: it says
	/// where what was drawn reaches without anybody working out a projection.
	fn top_row(image: &Image, background: [u8; 4]) -> Option<u32> {
		(0..image.height)
			.find(|row| (0..image.width).any(|x| distance(image.pixel(x, *row), background) > 20))
	}

	/// A bar of two cubes, the left one hung rigidly off bone zero and the
	/// right one pulled by whatever the caller says.
	///
	/// **Neither cube's center is the joint they turn about**, which the first
	/// two versions of this got wrong twice: a cube turned a quarter of a turn
	/// about its own center lands exactly on itself, so a picture of one
	/// cannot change whatever the shader does. The bone is at `x = 1`, the
	/// cubes are at `x = 0` and `x = 2`, and the right one is therefore in
	/// three different places depending on which bones pull it and how hard.
	///
	/// @param far - how much the right cube is pulled by bone zero and by
	/// bone one
	fn bar(far: [u8; 4]) -> MeshData {
		let mut data = MeshData::default();
		let pulls = [SkinVertex::rigid(0), SkinVertex { bones: [0, 1, 0, 0], weights: far }];

		for (along, pull) in [0.0_f32, 2.0].into_iter().zip(pulls) {
			let mut piece = mesh::cube();
			let base = u32::try_from(data.vertices.len()).expect("the bar is small");

			for vertex in &mut piece.vertices {
				vertex.position[0] += along;
			}

			data.indices
				.extend(piece.indices.iter().map(|index| index + base));
			data.skin
				.extend(std::iter::repeat_n(pull, piece.vertices.len()));
			data.vertices.extend(piece.vertices);
		}

		data
	}

	/// Two bones: one at the origin and one a unit along `x` from it, with the
	/// inverse binds worked out from the rests the way an importer does.
	fn elbow() -> SkeletonData {
		SkeletonData {
			bones: vec![
				Bone {
					name: "root".to_owned(),
					..Bone::default()
				},
				Bone {
					name: "arm".to_owned(),
					parent: 0,
					inverse_bind: Mat4::from_translation(Vec3::NEG_X),
					rest: Transform::at(Vec3::X),
				},
			],
		}
	}

	/// What the fixture's geometry is registered under.
	const BAR: &str = "bar";

	/// A world holding that bar, moved by that skeleton.
	///
	/// @param posed - whether to give it a pose at all
	/// @return the world, and the pose if there is one
	fn armed(posed: bool) -> (World, PoseId, MeshId) { armed_with(posed, [0, 255, 0, 0]) }

	/// The same, saying what pulls the right-hand cube.
	fn armed_with(posed: bool, far: [u8; 4]) -> (World, PoseId, MeshId) {
		let mut world = looking_world();
		let mesh = world.meshes.insert(BAR, bar(far));
		let skeleton = world.skeletons.insert("rig", elbow());
		let pose = if posed {
			world
				.poses
				.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)))
		} else {
			PoseId::NONE
		};
		// a unit left and a step back, so the three-unit bar sits across the
		// middle with room above it to swing into.
		let id = world
			.entities
			.spawn_at(Transform::at(Vec3::new(-1.0, -0.5, 0.0)));

		world
			.entities
			.set_renderable(id, Renderable::new(mesh, rgb(0.9, 0.7, 0.2)).posed(pose));

		(world, pose, mesh)
	}

	#[test]
	fn a_pose_two_entities_share_is_gathered_once_and_not_twice() {
		let Some(capture) = capture() else {
			return;
		};

		// the claim a picture cannot show: two entities of one character - a
		// model of two materials is exactly that - read one run of matrices.
		// Gathering per entity instead would draw the same thing and ask the
		// buffer for four times what it holds.
		let (world, pose, _) = armed(true);
		let mut joints =
			Joints::new(capture.scene.device()).expect("the joint buffer is described");

		joints.begin(&world);

		let first = joints.take(&world, pose);
		let second = joints.take(&world, pose);

		assert_ne!(first, skin::NO_JOINTS, "there is a run to share");
		assert_eq!(first, second, "and the second asker is told about the same one");
		assert_eq!(
			joints.len(),
			2,
			"which is two bones gathered once rather than two bones gathered twice"
		);
	}

	#[test]
	fn a_pose_that_is_gone_is_no_run_at_all() {
		let Some(capture) = capture() else {
			return;
		};

		let (mut world, pose, _) = armed(true);
		let mut joints =
			Joints::new(capture.scene.device()).expect("the joint buffer is described");

		world.poses.despawn(pose);
		joints.begin(&world);

		assert_eq!(
			joints.take(&world, pose),
			skin::NO_JOINTS,
			"a stale handle draws the shape it was modeled in rather than somebody else's pose"
		);
		assert_eq!(joints.len(), 0, "and nothing was gathered for it");
	}

	#[test]
	fn a_bone_that_turns_bends_the_geometry_hanging_off_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		let (mut world, pose, _) = armed(true);
		let straight = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let background = straight.pixel(1, 1);
		let before = top_row(&straight, background).expect("the bar is in the picture");

		// a quarter turn about z at the elbow, which swings the right-hand
		// cube up and over the joint.
		world.advance();
		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(1, Transform {
				rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
				..Transform::at(Vec3::X)
			});
		world.poses.snap_all();
		world.settle();

		let bent = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let after = top_row(&bent, background).expect("the bar is still in the picture");

		assert!(
			after + 20 < before,
			"the half hanging off the turned bone should reach higher up the picture, and the \
			 top went from row {before} to row {after}"
		);
	}

	#[test]
	fn a_vertex_two_bones_share_lands_where_neither_alone_would_put_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// one world and one bent pose, with the geometry swapped under it
		// three times: the right-hand cube pulled by all of bone zero, by all
		// of bone one, and by half of each. If the shader picked a bone rather
		// than adding the four, the third would be a copy of one of the first
		// two.
		//
		// @note: one world rather than three, and it matters. What tells a
		// capture a mesh changed is its registry slot's revision, and three
		// fresh worlds would each put their bar in slot one at revision zero -
		// so all three would be drawn with whichever arrived first. Rewriting
		// the entry under one name is the engine's own reload path and moves
		// the revision, which is exactly what has to happen.
		let (mut world, pose, _) = armed(true);

		world.advance();
		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(1, Transform {
				rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
				..Transform::at(Vec3::X)
			});
		world.poses.snap_all();
		world.settle();

		let mut shots = Vec::new();

		for far in [[255_u8, 0, 0, 0], [0, 255, 0, 0], [128, 127, 0, 0]] {
			world.meshes.insert(BAR, bar(far));
			shots.push(
				capture
					.shoot(&mut world)
					.expect("the capture renders"),
			);
		}

		assert_ne!(
			shots[0].pixels, shots[1].pixels,
			"the two bones really do put that cube in different places"
		);
		assert_ne!(
			shots[2].pixels, shots[0].pixels,
			"half of each is not all of the first, which is what picking rather than adding \
			 would have drawn"
		);
		assert_ne!(shots[2].pixels, shots[1].pixels, "and it is not all of the second either");
	}

	#[test]
	fn a_vertex_naming_a_bone_past_its_own_run_reads_the_last_one_instead() {
		let Some(mut capture) = capture() else {
			return;
		};

		// nothing upstream can quite rule this out: a mesh's block is checked
		// against the widest skeleton any file may have rather than against
		// the two-bone one moving it here, so an index of five reaches this
		// far. Clamped, it draws the last bone of its own run; trusted, it
		// reads whatever sits five matrices along - the next character's
		// shoulder, or nothing at all.
		let (mut world, pose, _) = armed(true);

		world.advance();
		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(1, Transform {
				rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
				..Transform::at(Vec3::X)
			});
		world.poses.snap_all();
		world.settle();

		let mut shots = Vec::new();

		for named in [1_u16, 5] {
			let mut geometry = bar([0, 255, 0, 0]);

			for vertex in geometry
				.skin
				.iter_mut()
				.filter(|vertex| vertex.bones[1] == 1)
			{
				vertex.bones[1] = named;
			}

			world.meshes.insert(BAR, geometry);
			shots.push(
				capture
					.shoot(&mut world)
					.expect("the capture renders"),
			);
		}

		assert_eq!(
			shots[0].pixels, shots[1].pixels,
			"an index past the end of the run is the end of the run, not a reach into the buffer"
		);
	}

	#[test]
	fn one_bent_character_does_not_bend_the_one_beside_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// two bars, each with a pose of its own, and only the second is bent.
		// If a run were addressed by anything but its own offset - the
		// distance to the end of the buffer, say - the still one would read
		// the bent one's matrices and move with it.
		let (mut world, _, mesh) = armed(true);
		let skeleton = world.skeletons.find("rig");
		let second = world
			.poses
			.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)));
		let id = world
			.entities
			.spawn_at(Transform::at(Vec3::new(-1.0, 1.5, 0.0)));

		world
			.entities
			.set_renderable(id, Renderable::new(mesh, rgb(0.2, 0.8, 0.9)).posed(second));

		let before = capture
			.shoot(&mut world)
			.expect("the capture renders");

		world.advance();
		world
			.poses
			.get_mut(second)
			.expect("the second pose is there")
			.set(1, Transform {
				rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
				..Transform::at(Vec3::X)
			});
		world.poses.snap_all();
		world.settle();

		let after = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let background = before.pixel(1, 1);
		// the still bar is the lower one and stops around two fifths down; the
		// bent one starts above that and only ever swings further up. So every
		// row below this belongs to the bar nobody touched, and measuring the
		// lot of them beats picking one and arguing about which.
		let below = SIZE.1 * 5 / 12;
		let same = (below..SIZE.1).all(|row| {
			(0..SIZE.0).all(|column| before.pixel(column, row) == after.pixel(column, row))
		});
		let anything = (below..SIZE.1).any(|row| {
			(0..SIZE.0).any(|column| distance(before.pixel(column, row), background) > 20)
		});

		assert_ne!(before.pixels, after.pixels, "the bent one really did move");
		assert!(anything, "there is a bar down there to be sure about");
		assert!(same, "and bending one pose left the geometry of the other exactly where it was");
	}

	#[test]
	fn a_resting_pose_draws_the_shape_the_mesh_was_modeled_in() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the same geometry twice: once moved by a pose every bone of which is
		// where its skeleton left it, and once by no pose at all. A resting
		// pose is the identity per bone, so the two are the same picture - and
		// if they are not, the inverse binds are being applied in the wrong
		// order or against the wrong bone.
		let (mut posed, ..) = armed(true);
		let (mut bare, ..) = armed(false);
		let with = capture
			.shoot(&mut posed)
			.expect("the capture renders");
		let without = capture
			.shoot(&mut bare)
			.expect("and so does the other");

		assert_eq!(with.pixels, without.pixels, "a resting pose moves nothing");
		assert!(
			top_row(&with, with.pixel(1, 1)).is_some(),
			"and there is something in the picture for that to be true of"
		);
	}

	#[test]
	fn an_entity_halfway_between_two_steps_is_drawn_halfway() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		let cube = world
			.entities
			.spawn_at(Transform::at(Vec3::new(-2.0, 0.0, 0.0)));
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		// one step's worth of movement, straight across the middle.
		world.advance();
		if let Some(transform) = world.entities.transform_mut(cube) {
			transform.position = Vec3::new(2.0, 0.0, 0.0);
		}
		world.settle();

		world.set_interpolation(0.5);
		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let center = shape_center(&image, SIZE.1 / 2, image.pixel(1, 1))
			.expect("the cube is somewhere in the picture");
		let middle = SIZE.0 / 2;

		assert!(
			center.abs_diff(middle) < 6,
			"the frame sits half a step past the first pose, so the cube belongs at column \
			 {middle} rather than {center}"
		);
	}

	#[test]
	fn an_entity_that_teleported_is_drawn_where_it_landed() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		let cube = world
			.entities
			.spawn_at(Transform::at(Vec3::new(-2.0, 0.0, 0.0)));
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		// the same move as above, and the same frame in the middle of the same
		// step. The only difference is that the game called it a teleport.
		world.advance();
		if let Some(transform) = world.entities.transform_mut(cube) {
			transform.position = Vec3::new(2.0, 0.0, 0.0);
		}
		world.entities.snap(cube);
		world.settle();

		world.set_interpolation(0.5);
		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let center = shape_center(&image, SIZE.1 / 2, image.pixel(1, 1))
			.expect("the cube is somewhere in the picture");
		let middle = SIZE.0 / 2;

		assert!(
			center > middle + 40,
			"a teleport is not smeared across the gap: the cube belongs at the far end, not at \
			 column {center}"
		);
	}

	#[test]
	fn a_capture_is_written_where_a_person_can_look_at_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// @note: a fixture, not the game's scene - the engine cannot depend on
		// the game crate. It is built to look like one: a floor, a cube in the
		// middle, a ring around it, lit from above and to one side.
		let mut world = World::new();
		plainly(&mut world);
		world.clear = rgb(0.04, 0.05, 0.07);
		world.ambient = Vec3::splat(0.22);
		world.light = Vec3::new(-0.5, -1.0, -0.35);
		world.camera.position = Vec3::new(4.0, 4.0, 7.0);
		world.camera.target = Vec3::ZERO;
		world.camera.fov_y = 0.9;

		let floor = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, -0.5, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(14.0, 1.0, 14.0),
		});
		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::QUAD, rgb(0.16, 0.17, 0.20)));

		let center = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		world
			.entities
			.set_renderable(center, Renderable::new(MeshId::CUBE, rgb(0.95, 0.76, 0.20)));

		for index in 0..8_u16 {
			let angle = f32::from(index) / 8.0 * std::f32::consts::TAU;
			let mut transform =
				Transform::at(Vec3::new(2.6 * angle.cos(), 0.0, 2.6 * angle.sin()));
			transform.rotation = Quat::from_rotation_y(angle);
			transform.set_scale(0.6);

			let cube = world.entities.spawn_at(transform);
			world.entities.set_renderable(
				cube,
				Renderable::new(
					MeshId::CUBE,
					rgb(f32::from(index).mul_add(-0.08, 0.9), 0.35, 0.6),
				),
			);
		}

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let path = std::env::temp_dir().join("colby-capture.png");
		image
			.write_png(&path)
			.expect("the png is written");
		crate::image::require_written(&path).expect("and is a plausible size");

		eprintln!("wrote {}", path.display());

		assert!(
			distance(image.pixel(SIZE.0 / 2, SIZE.1 * 2 / 3), image.pixel(1, 1)) > 20,
			"the scene is not empty"
		);
	}

	/// A square pyramid as OBJ: a base quad and four sides, no normals, wound
	/// counter-clockwise seen from outside. Nothing the engine can generate,
	/// which is the point of using it here.
	const PYRAMID_OBJ: &str = "\
v -1.0 -0.5 -1.0
v  1.0 -0.5 -1.0
v  1.0 -0.5  1.0
v -1.0 -0.5  1.0
v  0.0  1.0  0.0
f 1 2 3 4
f 2 1 5
f 3 2 5
f 4 3 5
f 1 4 5
";

	#[test]
	fn a_mesh_compiled_from_a_file_reaches_the_screen() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the whole pipeline, in order: text a person could have typed, through
		// the importer, out as bytes, onto disk, back off it, into the world's
		// registry, and finally onto a pixel this test looks at.
		let path = std::env::temp_dir().join("colby-capture-pyramid.cmesh");
		let imported = colby_asset::obj::import(PYRAMID_OBJ).expect("the source imports");
		std::fs::write(&path, colby_asset::encode(&imported).expect("it encodes"))
			.expect("and is written");

		let file = colby_asset::MeshFile::open(&path).expect("and reads back");

		assert_eq!(file.to_mesh_data(), imported, "unchanged by the trip through the file");

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let mesh = world
			.meshes
			.insert("test/pyramid", file.to_mesh_data());

		assert!(mesh.is_some(), "the registry took it");
		assert_ne!(mesh, MeshId::CUBE, "and gave it a slot of its own");

		let entity = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		world
			.entities
			.set_renderable(entity, Renderable::new(mesh, rgb(0.9, 0.1, 0.1)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(dominant(middle), 0, "the pyramid is on screen, in red: {middle:?}");
		assert!(middle[0] > 120, "and lit rather than a silhouette: {middle:?}");

		// its apex is at +y and its base stops at -0.5, so the top of the frame
		// is empty and the bottom of the shape is not. Get the import's
		// handedness wrong and this is the assertion that says so.
		let above = image.pixel(SIZE.0 / 2, 2);

		assert!(distance(above, image.pixel(1, 1)) <= 1, "nothing above the apex: {above:?}");

		drop(std::fs::remove_file(&path));
	}

	/// A world with one big triangle turned so that its edge crosses the
	/// middle of the picture at an angle no pixel grid agrees with.
	///
	/// A diagonal on purpose: a vertical or a horizontal edge lands on the
	/// grid and is not aliased at any sample count, so it would say nothing
	/// about either.
	fn a_slanted_edge() -> World {
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let mesh = world.meshes.insert(
			"test/slant",
			colby_asset::obj::import("v -4.0 -1.7 0\nv 4.0 -6.0 0\nv 4.0 2.3 0\nf 1 2 3\n")
				.expect("the slanted triangle imports"),
		);
		let entity = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		world
			.entities
			.set_renderable(entity, Renderable::new(mesh, rgb(0.9, 0.9, 0.9)));

		world
	}

	/// How many pixels down the middle column are neither the triangle nor the
	/// background but something in between.
	///
	/// The whole of what anti-aliasing produces and the whole of what its
	/// absence cannot: a hard edge has no partial pixels at all, whatever
	/// resolution it is drawn at.
	fn part_covered(image: &Image) -> usize {
		(0..SIZE.1)
			.map(|y| image.pixel(SIZE.0 / 2, y)[0])
			.filter(|red| *red > 20 && *red < 235)
			.count()
	}

	#[test]
	fn a_slanted_edge_has_pixels_between_the_two_sides_of_it_and_a_hard_one_has_none() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = a_slanted_edge();
		// off first, so what the picture looks like without any is measured on
		// the same device and the same triangle rather than remembered
		world.cvars.var(MSAA, Value::Float(1.0), "");

		let hard = part_covered(
			&capture
				.shoot(&mut world)
				.expect("the capture renders"),
		);

		world.cvars.set(MSAA, "4");

		let smooth = part_covered(
			&capture
				.shoot(&mut world)
				.expect("the capture renders again"),
		);

		assert_eq!(hard, 0, "a hard edge is one side or the other and nothing between");
		assert!(
			smooth > 0,
			"and four samples a pixel put something between them: {smooth} pixels"
		);
	}

	#[test]
	fn turning_it_off_again_puts_the_hard_edge_back() {
		// the half that says the rebuild goes both ways: eight pipelines and
		// two targets are replaced each time, and a pass whose attachments
		// disagreed with its pipeline would be a validation error rather than
		// a picture anybody could look at.
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = a_slanted_edge();
		let mut seen = Vec::new();

		for asked in ["4", "1", "4", "1"] {
			world.cvars.var(MSAA, Value::Float(1.0), "");
			world.cvars.set(MSAA, asked);
			seen.push(part_covered(
				&capture
					.shoot(&mut world)
					.expect("the capture renders"),
			));
		}

		assert!(seen[0] > 0 && seen[2] > 0, "on is on both times: {seen:?}");
		assert_eq!((seen[1], seen[3]), (0, 0), "and off is off both times: {seen:?}");
		assert_eq!(seen[0], seen[2], "and the same picture comes back: {seen:?}");
	}

	#[test]
	fn replacing_a_mesh_in_the_registry_changes_what_is_drawn() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let big = colby_asset::obj::import(PYRAMID_OBJ).expect("the source imports");
		let mesh = world.meshes.insert("test/swap", big);
		let entity = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		world
			.entities
			.set_renderable(entity, Renderable::new(mesh, rgb(0.9, 0.1, 0.1)));

		let before = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(dominant(before), 0, "the pyramid covers the middle: {before:?}");

		// the same name, so the same handle and the same entity - only the
		// geometry behind it moves. This is what editing a file under `assets/`
		// does to a running process.
		// small, and *clear of the middle pixel* rather than merely small: the
		// old triangle stopped a fiftieth of a unit short of the origin, which is
		// inside the pixel this test reads, and with four samples a pixel one of
		// them caught its top edge and tinted the answer. What the fixture always
		// meant is that the geometry is no longer where the test looks.
		let tiny =
			colby_asset::obj::import("v -0.02 -0.12 0\nv 0.02 -0.12 0\nv 0.0 -0.10 0\nf 1 2 3\n")
				.expect("the replacement imports");
		let again = world.meshes.insert("test/swap", tiny);

		assert_eq!(again, mesh, "the handle survived, so nothing had to be told");

		let after = capture
			.shoot(&mut world)
			.expect("the capture renders again")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert!(
			distance(after, image_corner(&mut capture, &mut world)) <= 1,
			"the new geometry reached the GPU, so the middle is clear again: {after:?}"
		);
	}

	/// The clear color, read from a corner of the current frame.
	fn image_corner(capture: &mut Capture, world: &mut World) -> [u8; 4] {
		capture
			.shoot(world)
			.expect("the capture renders")
			.pixel(1, 1)
	}

	#[test]
	fn a_shader_that_does_not_compile_leaves_the_picture_alone() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let cube = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		let before = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		let error = capture
			.scene_mut()
			.set_shader("this is not wgsl")
			.expect_err("nor is it going to become wgsl");

		assert!(!error.to_string().is_empty(), "and wgpu says why: {error}");

		let after = capture
			.shoot(&mut world)
			.expect("the capture still renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(
			before, after,
			"a bad edit costs a message, not the picture: {before:?} became {after:?}"
		);
	}

	#[test]
	fn a_shader_that_does_compile_replaces_the_picture() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let cube = world.entities.spawn_at(Transform::at(Vec3::ZERO));
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		// the real shader with one line changed: every fragment comes out blue,
		// whatever the entity asked for. Decisive, and it proves the swap
		// reached the GPU rather than merely being accepted.
		//
		// The guard is on the *anchor*, not on the result. It used to be on
		// the result, which is a guard that passes when the line has moved and
		// nothing was replaced at all - and that is exactly what happened the
		// day the last line of `shade` grew a fog.
		let shader = include_str!("shader.wgsl");
		let anchor = "return fogged(color, input.world_position);";

		assert!(
			anchor.len() > 1 && shader.contains(anchor),
			"the line this test edits has moved"
		);

		let source = shader.replace(anchor, "return vec3<f32>(0.0, 0.0, 1.0);");

		capture
			.scene_mut()
			.set_shader(&source)
			.expect("the edited shader compiles");

		let middle = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(dominant(middle), 2, "the cube is blue now, not red: {middle:?}");
	}

	/// An eight-by-eight image in four quarters: red, green over blue, white,
	/// with its whole chain.
	///
	/// Built by hand rather than decoded, because the decoder is
	/// `colby_asset`'s to test. What this one is for is the other end: whether
	/// those texels come out of the screen where the coordinates say they
	/// should. Eight across rather than two so that a sample a quarter of the
	/// way in sits two texels from any change of color: a software
	/// rasterizer's anisotropic filter was measured to blend in a fifth of a
	/// texel from beside the one sampled, which at two across is a fifth of
	/// the neighboring color and at eight across is more of the same one.
	fn quadrants() -> TextureData {
		const SIDE: u32 = 8;
		const HALF: u32 = SIDE / 2;
		let colors: [[u8; 4]; 4] = [
			[0xFF, 0x00, 0x00, 0xFF], // red
			[0x00, 0xFF, 0x00, 0xFF], // green
			[0x00, 0x00, 0xFF, 0xFF], // blue
			[0xFF, 0xFF, 0xFF, 0xFF], // white
		];

		let mut base = Vec::new();
		for y in 0..SIDE {
			for x in 0..SIDE {
				let quarter = usize::from(x >= HALF) + 2 * usize::from(y >= HALF);
				base.extend_from_slice(&colors[quarter]);
			}
		}

		TextureData {
			width: SIDE,
			height: SIDE,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: colby_asset::texture::build_chain(SIDE, SIDE, base, Texel::Rgba8Srgb)
				.expect("the chain builds"),
		}
	}

	#[test]
	fn a_texture_reaches_the_screen_the_way_up_its_coordinates_say() {
		// square, so that a quarter of the frame is the same distance in world
		// units across as it is down and the arithmetic below has one number
		// in it rather than two.
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};

		// through the file format, so this covers the bytes the compiler writes
		// and not only the registry.
		let bytes = colby_asset::texture::encode(&quadrants()).expect("it encodes");
		let file =
			colby_asset::TextureFile::from_bytes(colby_asset::AlignedBytes::from_slice(&bytes))
				.expect("and reads back");

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		// straight down, nudged off the pole where the view matrix is undefined.
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let texture = world
			.textures
			.insert("test/quadrants", file.to_texture_data());
		let material = world
			.materials
			.insert("test/quadrants", Material::textured(texture));

		// sized so that a quarter of the frame is a quarter of the quad, which
		// puts each sample below in the middle of a quarter of the image: two
		// texels from the nearest other color and from the edge the sampler
		// wraps around to. Nearer, and the test would be measuring the filter
		// rather than the coordinates - @ref `quadrants`.
		let across = (world.camera.fov_y / 2.0).tan() * HEIGHT * 2.0;
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		// looking down from just behind the origin, the screen's top is -z and
		// its left is -x. The quad's own coordinates put uv (0, 0) at -x, -z,
		// which is where the image's first texel belongs.
		let quarters = [
			((SQUARE / 4, SQUARE / 4), 0, "the first texel is at the top left"),
			((SQUARE * 3 / 4, SQUARE / 4), 1, "the second is to the right of it"),
			((SQUARE / 4, SQUARE * 3 / 4), 2, "the third is below the first"),
		];

		for ((x, y), channel, why) in quarters {
			let pixel = image.pixel(x, y);

			assert_eq!(dominant(pixel), channel, "{why}: {pixel:?}");
		}

		// the fourth is white, which has no dominant channel to check.
		let white = image.pixel(SQUARE * 3 / 4, SQUARE * 3 / 4);

		assert!(
			white[0].abs_diff(white[1]) < 12 && white[1].abs_diff(white[2]) < 12,
			"the fourth texel is white, so no channel wins: {white:?}"
		);
		assert!(white[0] > 200, "and it is bright: {white:?}");
	}

	/// A two-by-two picture in one color, half of whose texels are holes.
	///
	/// One color rather than four, so that what a test over this measures is
	/// the alpha and not which texel a sample landed on. The two alphas
	/// straddle the cutoff by a wide margin on both sides, so nothing here
	/// depends on where exactly the cutoff sits - only that it is between them.
	fn holed() -> TextureData { holed_in([0xFF, 0x22, 0x22]) }

	/// The same, in a color of somebody's choosing.
	///
	/// Only a test that puts the picture *over* something needs this: what a
	/// mask does is measured against the clear color, and what blending does
	/// has to be measured against whatever is behind it, which then has to be a
	/// different color from the picture or neither can be told from the other.
	fn holed_in(color: [u8; 3]) -> TextureData {
		const SOLID: u8 = 0xFF;
		const HOLE: u8 = 0x11;
		let base = vec![
			color[0], color[1], color[2], SOLID, // near
			color[0], color[1], color[2], HOLE, //
			color[0], color[1], color[2], HOLE, //
			color[0], color[1], color[2], SOLID, //
		];

		TextureData {
			width: 2,
			height: 2,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: colby_asset::texture::build_chain(2, 2, base, Texel::Rgba8Srgb)
				.expect("the chain builds"),
		}
	}

	/// Shoots one square frame of a floor quad wearing [`holed`], in a mode.
	///
	/// The camera is overhead and the quad is sized so that each quarter of the
	/// frame is the middle of one texel, which is the arrangement
	/// [`a_texture_reaches_the_screen_the_way_up_its_coordinates_say`] works
	/// out and for the same reason: off center the sampler blends with the
	/// neighbor it wraps around to, and the test would be measuring the filter.
	///
	/// @param blend - how the material reads the picture's alpha
	/// @return the frame, or `None` on a machine with no GPU
	fn shot_of(blend: Blend) -> Option<Image> {
		let mut capture = capture_of(SQUARE, SQUARE)?;

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let texture = world.textures.insert("test/holed", holed());
		let material = world
			.materials
			.insert("test/holed", Material { blend, ..Material::textured(texture) });

		let across = (world.camera.fov_y / 2.0).tan() * HEIGHT * 2.0;
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		Some(
			capture
				.shoot(&mut world)
				.expect("the capture renders"),
		)
	}

	#[test]
	fn a_masked_material_leaves_the_holes_in_its_picture_unfilled() {
		let (Some(solid), Some(cut)) = (shot_of(Blend::Opaque), shot_of(Blend::Mask)) else {
			return;
		};

		// the two texels whose alpha is high, and the two whose alpha is low.
		// The picture is one color, so what tells these apart is nothing but
		// the mode the material was drawn in.
		let kept = [(SQUARE / 4, SQUARE / 4), (SQUARE * 3 / 4, SQUARE * 3 / 4)];
		let holes = [(SQUARE * 3 / 4, SQUARE / 4), (SQUARE / 4, SQUARE * 3 / 4)];

		for (x, y) in kept {
			let (before, after) = (solid.pixel(x, y), cut.pixel(x, y));

			assert_eq!(dominant(after), 0, "a kept texel is the picture's red: {after:?}");
			assert!(after[0] > 120, "and it is lit: {after:?}");
			// the half that says the discard is about the alpha rather than
			// about the pipeline: a shader that threw every fragment away
			// would pass every assertion below and fail this one.
			assert!(
				distance(before, after) <= 1,
				"and masking changed nothing where there is no hole: {before:?} became {after:?}"
			);
		}

		// the quad is sized to fill the frame, so there is no corner of clear
		// color to compare against - and none is needed: the world clears to a
		// blue and every texel of the picture is a red, so which channel wins
		// says which of the two is there.
		for (x, y) in holes {
			let (before, after) = (solid.pixel(x, y), cut.pixel(x, y));

			assert_eq!(dominant(before), 0, "the same texel drawn solid is the red: {before:?}");
			assert_eq!(
				dominant(after),
				2,
				"and drawn masked there is nothing there but the clear: {after:?}"
			);
			// what makes the pair above a measurement rather than two
			// coincidences: a build that never reached the masked pipeline
			// would hand back one picture twice.
			assert!(
				distance(before, after) > 20,
				"so the two modes really do differ here: {before:?} against {after:?}"
			);
		}
	}

	#[test]
	fn a_metal_and_a_dielectric_of_the_same_color_do_not_look_the_same() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(0.2);
		world.light = Vec3::new(-0.4, -0.6, -1.0);

		let plastic = world
			.materials
			.insert("test/plastic", Material::DEFAULT.finished(0.0, 0.5));
		let metal = world
			.materials
			.insert("test/metal", Material::DEFAULT.finished(1.0, 0.25));

		let left = world
			.entities
			.spawn_at(Transform::at(Vec3::new(-1.6, 0.0, 0.0)));
		world
			.entities
			.set_renderable(left, Renderable::of(MeshId::CUBE, plastic, rgb(0.8, 0.7, 0.3)));

		let right = world
			.entities
			.spawn_at(Transform::at(Vec3::new(1.6, 0.0, 0.0)));
		world
			.entities
			.set_renderable(right, Renderable::of(MeshId::CUBE, metal, rgb(0.8, 0.7, 0.3)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let dielectric = image.pixel(SIZE.0 / 4, SIZE.1 / 2);
		let metallic = image.pixel(SIZE.0 * 3 / 4, SIZE.1 / 2);

		assert!(distance(dielectric, image.pixel(1, 1)) > 20, "both cubes are on screen");
		assert!(distance(metallic, image.pixel(1, 1)) > 20, "both of them");

		// a metal has no diffuse term, so under one light and a little ambient
		// it comes out darker than the same color scattering. If the material
		// never reached the instance data these would be the same pixel.
		assert!(
			distance(dielectric, metallic) > 20,
			"the same color made of two things looks like two things: {dielectric:?} against \
			 {metallic:?}"
		);
	}

	#[test]
	fn a_floor_seen_from_above_is_not_culled_away() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(8.0, 1.0, 8.0),
		});
		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::QUAD, rgb(0.1, 0.9, 0.1)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(
			dominant(middle),
			1,
			"the quad is wound facing up, so looking down at it shows its front: {middle:?}"
		);
	}

	/// Whether a green segment was drawn across the middle of the picture.
	///
	/// A short column rather than one pixel: a segment is one pixel wide, the
	/// middle of a two-hundred-and-forty-row image is the boundary between two
	/// of them, and which one a rasterizer picks is not something a test should
	/// have an opinion about.
	/// Whether the segment in this world reaches the middle of the picture.
	///
	/// **Asked as a comparison against the same world with no segment in it**,
	/// and the reason is anti-aliasing. A segment one pixel wide that lands
	/// between two rows of samples is drawn as two half-covered rows, and half
	/// a green line over a red cube is a grey pixel - no threshold on one
	/// pixel of one row can tell that from the cube on its own, and a
	/// threshold that could would be asking where the samples happened to
	/// fall. What the picture does conserve is how much *more* green there is
	/// down the column with the line than without it, and that is this.
	///
	/// @param capture - the same capture, so both pictures are the same device
	/// @param world - the world with the segment in it
	/// @param bare - the same world with none
	fn green_across_the_middle(
		capture: &mut Capture,
		world: &mut World,
		bare: &mut World,
	) -> bool {
		green_gain(capture, world, bare) > MARGIN
	}

	/// How much more green a world's middle column has than the same world
	/// with no segment in it.
	///
	/// @param capture - the same capture, so both pictures are one device
	/// @param world - the world with the segment in it
	/// @param bare - the same world with none
	fn green_gain(capture: &mut Capture, world: &mut World, bare: &mut World) -> u32 {
		let lined = green_down_the_middle(&capture.shoot(world).expect("the capture renders"));
		let plain = green_down_the_middle(
			&capture
				.shoot(bare)
				.expect("the capture renders the second time"),
		);

		lined.saturating_sub(plain)
	}

	/// How much green there is down the middle column, over the rows a debug
	/// segment could be on.
	fn green_down_the_middle(image: &Image) -> u32 {
		let middle = SIZE.1 / 2;

		(middle - 6..=middle + 6)
			.map(|y| u32::from(image.pixel(SIZE.0 / 2, y)[1]))
			.sum()
	}

	/// How much more green down the middle column counts as a segment.
	///
	/// **Small on purpose, and it is the sample count that makes it so**: a
	/// segment one pixel wide covers a fraction of the samples of the two rows
	/// it falls between, so what it adds over a lit red cube is tens rather
	/// than hundreds. The case it has to be told apart from adds exactly
	/// nothing - a segment the depth buffer threw away leaves the column bit
	/// for bit as it was - so any margin above the noise separates them, and
	/// this one is measured to be about half of the smallest real gain.
	const MARGIN: u32 = 25;

	/// A world holding one debug segment along x, at a given depth.
	fn with_a_line(at_z: f32, on_top: bool) -> World { lined(at_z, on_top, true) }

	/// The same world with no segment in it: what every one of these is
	/// measured against.
	fn with_no_line() -> World { lined(0.0, false, false) }

	/// One of the two, so that the only difference between them is the line.
	fn lined(at_z: f32, on_top: bool, drawn: bool) -> World {
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let (from, to) = (Vec3::new(-3.0, 0.0, at_z), Vec3::new(3.0, 0.0, at_z));
		let green = rgb(0.1, 0.9, 0.1);

		if !drawn {
			return world;
		}

		if on_top {
			world.debug.on_top().line(from, to, green);
		} else {
			world.debug.line(from, to, green);
		}

		world
	}

	/// A red cube two units across, standing at the origin.
	fn blocking(world: &mut World) {
		let cube = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));
	}

	/// A world with a sky of three flat, unmistakable colors.
	///
	/// Not a plausible sky: red up, green across, blue down, so that which
	/// band a pixel came out of is one comparison rather than an argument
	/// about a gradient.
	fn skied_world() -> World {
		let mut world = looking_world();
		world.clear = rgb(0.0, 0.0, 0.0);
		world.sky = Sky::gradient(rgb(1.0, 0.0, 0.0), rgb(0.0, 1.0, 0.0), rgb(0.0, 0.0, 1.0));

		world
	}

	/// A world that is nothing but a flat surface of about this brightness.
	///
	/// Two things about it are what make a meter testable. The sun is turned
	/// to travel *towards* the camera, so the face pointing at it catches none
	/// of it and what is in front of the lens is the ambient times the surface
	/// rather than the sum of two terms. And the surface fills the frame: a
	/// meter reads the whole picture, so a bright thing on a black background
	/// measures as a dark picture and asks for the ceiling whatever the thing
	/// is - which is correct, and is not what a test of the ceiling wants.
	fn glowing(level: f32) -> World {
		let mut world = looking_world();

		world.post = Post::DEFAULT;
		world.clear = rgb(0.0, 0.0, 0.0);
		world.light = Vec3::Z;
		world.ambient = Vec3::splat(level);

		let wall = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(40.0, 40.0, 1.0),
		});
		world
			.entities
			.set_renderable(wall, Renderable::new(MeshId::CUBE, rgb(1.0, 1.0, 1.0)));

		world
	}

	/// The green of the middle pixel, which is the one that moves most per
	/// unit of light.
	fn middle(image: &Image) -> u32 { u32::from(image.pixel(SIZE.0 / 2, SIZE.1 / 2)[1]) }

	#[test]
	fn a_curve_and_no_curve_are_not_the_same_picture() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = glowing(0.5);
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;

		world.post.tonemap = ToneMap::None;
		let plain = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.tonemap = ToneMap::Aces;
		let filmic = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.tonemap = ToneMap::Reinhard;
		let simple = middle(&capture.shoot(&mut world).expect("it renders"));

		assert!(plain > 0 && plain < 255, "a half-lit surface is neither black nor white");
		assert_ne!(filmic, plain, "and the filmic curve moves it");
		assert_ne!(simple, plain, "and so does the cheap one");
		assert_ne!(simple, filmic, "and the two curves are not each other");
	}

	#[test]
	fn without_a_curve_two_different_brightnesses_past_white_are_the_same_white() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the whole of what the float target bought, in one comparison. Before
		// it, the target was eight bits and everything past one was the same
		// white; a curve with a shoulder keeps them apart.
		let mut dim = glowing(2.0);
		dim.post.auto_exposure = false;
		dim.post.exposure = 1.0;
		let mut blazing = glowing(8.0);
		blazing.post.auto_exposure = false;
		blazing.post.exposure = 1.0;

		dim.post.tonemap = ToneMap::None;
		blazing.post.tonemap = ToneMap::None;

		assert_eq!(middle(&capture.shoot(&mut dim).expect("it renders")), 255);
		assert_eq!(
			middle(&capture.shoot(&mut blazing).expect("it renders")),
			255,
			"clamped, both of them, which is what an eight-bit target always did"
		);

		dim.post.tonemap = ToneMap::Aces;
		blazing.post.tonemap = ToneMap::Aces;

		let softer = middle(&capture.shoot(&mut dim).expect("it renders"));
		let brighter = middle(&capture.shoot(&mut blazing).expect("it renders"));

		assert!(softer < 255, "the curve has a shoulder, so twice white is not white: {softer}");
		assert!(brighter > softer, "and eight times is brighter than twice: {brighter}");
	}

	#[test]
	fn the_exposure_scales_the_picture_before_the_curve() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = glowing(0.2);
		world.post.auto_exposure = false;
		world.post.tonemap = ToneMap::None;

		world.post.exposure = 1.0;
		let one = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.exposure = 2.0;
		let two = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.exposure = 0.5;
		let half = middle(&capture.shoot(&mut world).expect("it renders"));

		assert!(half < one && one < two, "{half} then {one} then {two}");
	}

	#[test]
	fn an_eye_opens_already_adapted_and_lifts_a_dark_room() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the whole of why a process that renders exactly one frame gets a
		// picture worth looking at. A fresh capture has no history, so the eye
		// is *set* to what it measures rather than moved towards it.
		let mut world = glowing(0.02);
		world.post.tonemap = ToneMap::None;

		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
		let unaided = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.auto_exposure = true;
		let adapted = middle(&capture.shoot(&mut world).expect("it renders"));

		assert!(
			adapted > unaided.saturating_mul(2),
			"a dark room is opened up rather than left dark: {adapted} against {unaided}"
		);
	}

	#[test]
	fn an_eye_stops_down_in_a_bright_room_and_stays_inside_its_two_numbers() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = glowing(4.0);
		world.post.tonemap = ToneMap::None;
		world.post.auto_exposure = true;
		let stopped = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
		let wide = middle(&capture.shoot(&mut world).expect("it renders"));

		assert_eq!(wide, 255, "four times white without a meter is white");
		assert!(stopped < wide, "and with one it is stopped down: {stopped}");

		// and the floor holds: a room this bright would ask for a twentieth,
		// and the smallest the eye may be is a twentieth, so the two agree.
		let mut held = glowing(4.0);
		held.post.tonemap = ToneMap::None;
		held.post.exposure_min = 1.0;
		held.post.exposure_max = 1.0;

		assert_eq!(
			middle(&capture.shoot(&mut held).expect("it renders")),
			255,
			"an eye held at one is an eye that is not metering"
		);
	}

	#[test]
	fn the_bias_moves_a_measured_exposure_by_stops() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = glowing(0.1);
		world.post.tonemap = ToneMap::None;

		let level = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.exposure_bias = -2.0;
		let darker = middle(&capture.shoot(&mut world).expect("it renders"));

		assert!(darker < level, "two stops down is darker: {darker} against {level}");
	}

	#[test]
	fn distance_fades_a_surface_towards_the_fog() {
		let Some(mut capture) = capture() else {
			return;
		};

		// one wall filling the frame and a fog that is unmistakably red. What
		// changes between the three readings is how far the wall is and how
		// thick the air is, and nothing else - which is what makes the two
		// comparisons about the falloff rather than about the geometry.
		let mut world = glowing(0.8);
		world.post.tonemap = ToneMap::None;
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
		world.post.fog = rgb(1.0, 0.0, 0.0);

		let clear = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.fog_density = 0.02;
		let thin = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.fog_density = 0.6;
		let thick = middle(&capture.shoot(&mut world).expect("it renders"));

		// green, because the fog has none: the more fog there is the less
		// green comes back.
		assert!(clear > 200, "the wall alone is bright: {clear}");
		assert!(thin > clear - 10, "five units of thin air changes almost nothing: {thin}");
		assert!(thick < 25, "and thick air takes almost all of it: {thick}");

		// the same air, further away
		world.post.fog_density = 0.1;
		let near = middle(&capture.shoot(&mut world).expect("it renders"));
		world.camera.position = Vec3::new(0.0, 0.0, 18.0);
		let far = middle(&capture.shoot(&mut world).expect("it renders"));

		assert!(
			far + 40 < near,
			"the same air over three times the distance takes far more: {far} against {near}"
		);
	}

	#[test]
	fn no_density_is_no_fog_at_all_and_not_a_faint_one() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the branchless arithmetic in one assertion: `exp(0)` is exactly one,
		// so a density of nought has to leave the picture untouched rather
		// than nearly so - even with a fog color as loud as this.
		let mut world = glowing(0.4);
		world.post.tonemap = ToneMap::None;
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;

		let plain = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.fog = rgb(1.0, 0.0, 1.0);
		world.post.fog_density = 0.0;

		assert_eq!(
			middle(&capture.shoot(&mut world).expect("it renders")),
			plain,
			"a fog nobody asked for is not a fog anybody gets"
		);
	}

	#[test]
	fn nothing_glows_until_something_asks_it_to() {
		let Some(mut capture) = capture() else {
			return;
		};

		// a small bright thing on a dark ground, which is the shape bloom is
		// for and the shape it is easiest to be wrong about.
		let mut world = looking_world();
		world.post = Post::DEFAULT;
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
		world.post.tonemap = ToneMap::None;
		world.ambient = Vec3::splat(6.0);
		world.light = Vec3::Z;

		let lump = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(0.7),
		});
		world
			.entities
			.set_renderable(lump, Renderable::new(MeshId::CUBE, rgb(1.0, 1.0, 1.0)));

		// a point well clear of the lump, where only a glow could reach
		let beside = (SIZE.0 / 2 + 40, SIZE.1 / 2);
		let dark = capture.shoot(&mut world).expect("it renders");

		world.post.bloom = 1.0;
		let glowing = capture.shoot(&mut world).expect("it renders");

		let before = u32::from(dark.pixel(beside.0, beside.1)[1]);
		let after = u32::from(glowing.pixel(beside.0, beside.1)[1]);

		assert!(before < 40, "beside the lump is dark to begin with: {before}");
		assert!(after > before + 10, "and a glow reaches it: {after} against {before}");
	}

	#[test]
	fn a_threshold_decides_what_is_bright_enough_to_glow() {
		let Some(mut capture) = capture() else {
			return;
		};

		// a wall at about half, so that the reading has room to move up before
		// it clips and the test is measuring the threshold rather than a clamp
		let mut world = glowing(0.5);
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
		world.post.tonemap = ToneMap::None;

		let plain = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.bloom = 1.0;
		world.post.bloom_threshold = 0.1;
		let low = middle(&capture.shoot(&mut world).expect("it renders"));

		world.post.bloom_threshold = 40.0;
		let high = middle(&capture.shoot(&mut world).expect("it renders"));

		assert!(plain > 0 && plain < 255, "the wall alone is neither black nor white: {plain}");
		assert!(low > plain, "a wall past the threshold glows: {low} against {plain}");
		assert_eq!(high, plain, "and a threshold nothing reaches adds nothing at all");
	}

	#[test]
	fn a_world_with_no_sky_shows_the_clear_color_behind_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = skied_world();
		world.sky.kind = SkyKind::None;

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(
			image.pixel(SIZE.0 / 2, SIZE.1 / 2),
			[0, 0, 0, 255],
			"the word is what decides it, and the three colors are still on the record"
		);
	}

	#[test]
	fn a_sky_turned_on_is_the_three_colors_in_the_three_directions() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = skied_world();
		// looking at the origin from +z, so the middle row is the horizon, the
		// top of the picture is up and the bottom is down
		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(
			dominant(image.pixel(SIZE.0 / 2, 2)),
			0,
			"the top of the picture is the zenith"
		);
		assert_eq!(dominant(image.pixel(SIZE.0 / 2, SIZE.1 / 2)), 1, "the middle is the horizon");
		assert_eq!(
			dominant(image.pixel(SIZE.0 / 2, SIZE.1 - 3)),
			2,
			"and the bottom is the ground"
		);
	}

	#[test]
	fn a_sky_is_behind_the_world_rather_than_over_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the whole point of the depth test on the sky's pipeline: a cube in
		// front of the camera has to survive it. Without the test, or with the
		// sky drawn after the blended pass, this is a screen of sky.
		let mut world = skied_world();
		world.ambient = Vec3::splat(1.0);
		let cube = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.9, 0.9)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert!(
			middle[0] > 150 && middle[1] > 150 && middle[2] > 150,
			"the cube is in front of the sky and is white: {middle:?}"
		);
		assert_eq!(
			dominant(image.pixel(SIZE.0 / 2, 2)),
			0,
			"and the sky is still there where the cube is not"
		);
	}

	#[test]
	fn a_debug_segment_reaches_the_screen() {
		let Some(mut capture) = capture() else {
			return;
		};

		let (mut world, mut bare) = (with_a_line(0.0, false), with_no_line());

		assert!(
			green_across_the_middle(&mut capture, &mut world, &mut bare),
			"a segment through the origin, drawn by a camera aimed at the origin, crosses the \
			 middle of the picture"
		);
	}

	#[test]
	fn a_debug_segment_behind_something_is_hidden_by_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the segment is two units behind the origin and the cube spans one
		// either side of it, so the cube is squarely in the way.
		let (mut world, mut bare) = (with_a_line(-2.0, false), with_no_line());
		blocking(&mut world);
		blocking(&mut bare);

		assert!(
			!green_across_the_middle(&mut capture, &mut world, &mut bare),
			"the whole point of drawing inside the scene's pass is that the depth buffer applies"
		);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(
			dominant(image.pixel(SIZE.0 / 2, SIZE.1 / 2)),
			0,
			"and what is there instead is the cube"
		);
	}

	#[test]
	fn a_debug_segment_asked_for_on_top_ignores_what_is_in_front_of_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the same scene as the test above, with the one bit flipped.
		let (mut world, mut bare) = (with_a_line(-2.0, true), with_no_line());
		blocking(&mut world);
		blocking(&mut bare);

		assert!(
			green_across_the_middle(&mut capture, &mut world, &mut bare),
			"a contact normal starts on the surface that made it, so half of the debug drawing \
			 would be invisible without this"
		);
	}

	/// How bright a pixel is, as the sum of its three color channels.
	fn brightness(pixel: [u8; 4]) -> u32 {
		u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])
	}

	/// An eight-texel normal map: the left four lean towards `-x`, the right
	/// four towards `+x`, all by about forty-five degrees.
	///
	/// One level and no chain, so that what reaches the sampler is what is
	/// written here rather than an average of it; four texels a side rather
	/// than one for the reason `quadrants` gives.
	fn leaning_normals() -> TextureData {
		// tangent space, and the quad's tangent runs along +x. 37 and 217 are
		// -0.707 and +0.707 folded into a byte; 128 is zero.
		let left = [37, 128, 217, 255];
		let right = [217, 128, 217, 255];
		let base = [left; 4]
			.concat()
			.into_iter()
			.chain([right; 4].concat())
			.collect();

		TextureData {
			width: 8,
			height: 1,
			faces: 1,
			texel: Texel::Rgba8Unorm,
			levels: vec![base],
		}
	}

	#[test]
	fn a_normal_map_turns_the_light_a_surface_catches() {
		// square, for the reason the texture test is: the quad then fills the
		// frame in both directions, and a sample a quarter of the way across
		// lands in the middle of a half, two texels from the seam and from the
		// wrap rather than in the blend between two.
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};

		let mut world = looking_world();
		// down and towards -x, so a surface leaning into +x catches all of it
		// and one leaning into -x catches none. A flat surface is between.
		world.light = Vec3::new(-1.0, -1.0, 0.0).normalize();
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let normals = world
			.textures
			.insert("test/leaning", leaning_normals());
		let material = world
			.materials
			.insert("test/leaning", Material::DEFAULT.bumped(normals));

		let across = (world.camera.fov_y / 2.0).tan() * HEIGHT * 2.0;
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		let (middle, quarter) = (SQUARE / 2, SQUARE / 4);
		let (left, right) = (
			brightness(image.pixel(quarter, middle)),
			brightness(image.pixel(quarter * 3, middle)),
		);

		assert!(
			right > left * 3,
			"the half of the quad the map leans into the light is much brighter: {right} \
			 against {left}"
		);
		assert!(left < 60, "and the half it leans away is nearly out of the light: {left}");
	}

	/// How wide the view is at the floor, in world units.
	fn view_across(world: &World) -> f32 { (world.camera.fov_y / 2.0).tan() * HEIGHT * 2.0 }

	/// A green floor filling a square view from straight above, lit by the
	/// ambient alone, so that what a pixel is is what the surface is.
	fn floor_below() -> (World, EntityId) {
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let across = view_across(&world);
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::QUAD, rgb(0.1, 0.9, 0.1)));

		(world, floor)
	}

	/// A decal of this material thrown straight down with its top towards `-z`,
	/// its box this wide either way and a unit deep, its middle this far below
	/// the floor.
	fn thrown_down(
		world: &mut World,
		material: MaterialId,
		order: i32,
		side: f32,
		below: f32,
	) -> EntityId {
		let decal = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, -below, 0.0),
			rotation: Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2),
			scale: Vec3::new(side, side, 1.0),
		});

		world.entities.set_renderable(decal, Renderable {
			material,
			color: Vec3::ONE,
			..Renderable::NOTHING
		});
		world
			.entities
			.set_decal(decal, Decal { order, ..Decal::BOX });

		decal
	}

	/// A decal over the middle of the floor, half the view across and turned to
	/// throw straight down: the square in the middle of the picture from a
	/// quarter of the way across it to three quarters.
	fn daubed(world: &mut World, material: MaterialId, order: i32) -> EntityId {
		let half = view_across(world) * 0.5;

		thrown_down(world, material, order, half, 0.0)
	}

	/// A decal over the whole of the floor the view holds.
	fn covered(world: &mut World, material: MaterialId) -> EntityId {
		let across = view_across(world);

		thrown_down(world, material, 0, across, 0.0)
	}

	/// A flat red material, which is a decal throwing its tint alone.
	fn red(world: &mut World) -> MaterialId {
		world
			.materials
			.insert("test/red", Material::colored(rgb(1.0, 0.05, 0.05)))
	}

	#[test]
	fn a_decal_paints_the_floor_inside_its_box_and_nothing_outside_it() {
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut world, _) = floor_below();
		let paint = red(&mut world);
		daubed(&mut world, paint, 0);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let (inside, outside) =
			(image.pixel(SQUARE / 2, SQUARE / 2), image.pixel(SQUARE / 8, SQUARE / 2));

		assert_eq!(dominant(inside), 0, "the middle of the floor is under the decal: {inside:?}");
		assert_eq!(dominant(outside), 1, "and beside its box the floor is green: {outside:?}");
		assert_eq!(capture.scene_mut().drawn().decals, 1, "and the frame says it carried one");
	}

	#[test]
	fn a_frame_that_carries_no_decals_is_the_picture_there_would_be_without_any() {
		// the negative control as a test: the budget at nought and the decal left
		// out of the world are one picture to the byte
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut bare, _) = floor_below();
		let without = capture
			.shoot(&mut bare)
			.expect("the capture renders");

		let (mut world, _) = floor_below();
		let paint = red(&mut world);
		daubed(&mut world, paint, 0);
		world
			.cvars
			.var(crate::decal::DECALS, Value::Float(0.0), "");

		let none = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(none.pixels == without.pixels, "a frame that carries none paints nothing");

		world.cvars.set(crate::decal::DECALS, "1");
		let one = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(
			one.pixels != without.pixels,
			"and one it carries is seen, which says the two above are not alike by accident"
		);
	}

	#[test]
	fn a_floor_that_takes_no_decals_and_a_hidden_decal_both_leave_the_floor_alone() {
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut bare, _) = floor_below();
		let without = capture
			.shoot(&mut bare)
			.expect("the capture renders");

		let (mut refusing, floor) = floor_below();
		let paint = red(&mut refusing);
		daubed(&mut refusing, paint, 0);
		assert!(refusing.entities.set_takes_decals(floor, false));

		let refused = capture
			.shoot(&mut refusing)
			.expect("the capture renders");

		assert!(refused.pixels == without.pixels, "a floor that takes no decals is the floor");

		let (mut hiding, _) = floor_below();
		let paint = red(&mut hiding);
		let decal = daubed(&mut hiding, paint, 0);
		assert!(hiding.entities.set_hidden(decal, true));

		let hidden = capture
			.shoot(&mut hiding)
			.expect("the capture renders");

		assert!(hidden.pixels == without.pixels, "and a hidden decal paints nothing at all");
	}

	#[test]
	fn of_two_decals_over_one_point_the_one_of_the_higher_order_is_on_top() {
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut world, _) = floor_below();
		let paint = red(&mut world);
		let blue = world
			.materials
			.insert("test/blue", Material::colored(rgb(0.05, 0.05, 1.0)));
		let under = daubed(&mut world, paint, 0);
		daubed(&mut world, blue, 1);

		let first = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SQUARE / 2, SQUARE / 2);

		assert_eq!(dominant(first), 2, "the blue one, of the higher order, is on top: {first:?}");

		// the discriminating half: by slot alone the red one is always under
		assert!(
			world
				.entities
				.set_decal(under, Decal { order: 2, ..Decal::BOX })
		);
		let second = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SQUARE / 2, SQUARE / 2);

		assert_eq!(
			dominant(second),
			0,
			"and raising the red one's order puts it on top: {second:?}"
		);
	}

	#[test]
	fn a_decal_picture_lands_the_right_way_up_and_the_right_way_round() {
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut world, _) = floor_below();
		let picture = world
			.textures
			.insert("test/quadrants", quadrants());
		let material = world
			.materials
			.insert("test/quadrants", Material::textured(picture));
		daubed(&mut world, material, 0);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		// the decal covers the middle half of the picture, so the middle of each
		// quarter of it is three eighths or five eighths of the way across. Its
		// top, +y, is thrown towards -z, which from above is the top of the
		// screen, and its right edge is +x, which is the screen's right.
		let (near, far) = (SQUARE * 3 / 8, SQUARE * 5 / 8);
		let quarters = [
			((near, near), 0, "the picture's first texel is at the top left"),
			((far, near), 1, "the second is to the right of it"),
			((near, far), 2, "the third is below the first"),
		];

		for ((x, y), channel, why) in quarters {
			let pixel = image.pixel(x, y);

			assert_eq!(dominant(pixel), channel, "{why}: {pixel:?}");
		}

		let white = image.pixel(far, far);

		assert!(
			white[0].abs_diff(white[1]) < 12 && white[1].abs_diff(white[2]) < 12,
			"the fourth is white, so no channel wins: {white:?}"
		);
	}

	#[test]
	fn a_decal_normal_map_turns_the_light_the_floor_under_it_catches() {
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut world, floor) = floor_below();
		// down and towards -x, so a surface leaning into +x catches all of it and
		// one leaning into -x catches none, the floor's own map test's light
		world.light = Vec3::new(-1.0, -1.0, 0.0).normalize();
		world.ambient = Vec3::ZERO;
		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::QUAD, Vec3::ONE));

		let normals = world
			.textures
			.insert("test/leaning", leaning_normals());
		let material = world
			.materials
			.insert("test/leaning", Material::DEFAULT.bumped(normals));
		daubed(&mut world, material, 0);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let (left, right) = (
			brightness(image.pixel(SQUARE * 3 / 8, SQUARE / 2)),
			brightness(image.pixel(SQUARE * 5 / 8, SQUARE / 2)),
		);

		assert!(
			right > left * 3,
			"the half of the decal whose map leans into the light is much brighter: {right} \
			 against {left}"
		);
	}

	#[test]
	fn a_decal_fades_on_a_face_edge_on_to_it_by_as_much_as_it_says() {
		// a wall facing the camera, and a decal thrown straight down through it:
		// the wall's face is edge on to the way the picture is thrown
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let wall = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(4.0, 4.0, 0.1),
		});
		world
			.entities
			.set_renderable(wall, Renderable::new(MeshId::CUBE, rgb(0.1, 0.9, 0.1)));

		let paint = red(&mut world);
		let decal = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::from_rotation_x(-core::f32::consts::FRAC_PI_2),
			scale: Vec3::splat(2.0),
		});
		world
			.entities
			.set_renderable(decal, Renderable { material: paint, ..Renderable::NOTHING });
		world.entities.set_decal(decal, Decal::BOX);

		let faded = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(dominant(faded), 1, "edge on, the default fade paints nothing: {faded:?}");

		world
			.entities
			.set_decal(decal, Decal { fade: 0.0, ..Decal::BOX });
		let painted = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(dominant(painted), 0, "and a fade of nought paints it anyway: {painted:?}");
	}

	#[test]
	fn a_decal_fades_towards_the_two_faces_its_picture_is_thrown_between() {
		// the floor through the middle of the box, and then nine tenths of the
		// way from the middle to the face above it: still painted there, and
		// visibly less, so that a decal does not end in a hard line on
		// something poking through one of those faces
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let shoot = |capture: &mut Capture, below: f32| {
			let (mut world, _) = floor_below();
			let paint = red(&mut world);
			let half = view_across(&world) * 0.5;
			thrown_down(&mut world, paint, 0, half, below);

			capture
				.shoot(&mut world)
				.expect("the capture renders")
				.pixel(SQUARE / 2, SQUARE / 2)
		};

		let middle = shoot(&mut capture, 0.0);
		// the box is a unit deep, so a middle 0.45 below the floor leaves the
		// floor 0.45 of the way from it to the face that looks up
		let near_a_face = shoot(&mut capture, 0.45);

		assert_eq!(dominant(middle), 0, "through the middle the decal paints: {middle:?}");
		assert_eq!(dominant(near_a_face), 0, "and near a face it still does: {near_a_face:?}");
		assert!(
			near_a_face[1] > middle[1] + 40,
			"but the floor's green comes back through it there: {near_a_face:?} against \
			 {middle:?}"
		);
	}

	/// A normal map every texel of which leans down the picture by forty-five
	/// degrees: towards where `v` grows, on a face and in a decal's picture
	/// alike.
	fn leaning_down_the_picture() -> TextureData {
		const SIDE: u32 = 4;
		let mut base = Vec::new();
		for _ in 0..SIDE * SIDE {
			// 217 is +0.707 folded into a byte and 128 is zero, as in
			// `leaning_normals`
			base.extend_from_slice(&[128, 217, 217, 255]);
		}

		TextureData {
			width: SIDE,
			height: SIDE,
			faces: 1,
			texel: Texel::Rgba8Unorm,
			levels: colby_asset::texture::build_chain(SIDE, SIDE, base, Texel::Rgba8Unorm)
				.expect("a chain of four texels a side builds"),
		}
	}

	#[test]
	fn a_map_thrown_by_a_decal_turns_a_floor_as_the_same_map_laid_on_it_does() {
		// the quad's v grows towards +z, and so does the v of a picture thrown
		// straight down with its top towards -z. A light falling towards -z
		// catches a surface leaning into +z better than a flat one, so the map
		// laid on the floor lights it more than no map does, and thrown onto a
		// plain floor by a decal it has to light it the same
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let shoot = |capture: &mut Capture, laid: bool, thrown: bool| {
			let mut world = looking_world();
			world.light = Vec3::new(0.0, -1.0, -2.0).normalize();
			world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

			let normals = world
				.textures
				.insert("test/down", leaning_down_the_picture());
			let bumped = world
				.materials
				.insert("test/down", Material::DEFAULT.bumped(normals));

			let across = view_across(&world);
			let floor = world.entities.spawn_at(Transform {
				position: Vec3::ZERO,
				rotation: Quat::IDENTITY,
				scale: Vec3::new(across, 1.0, across),
			});
			let look = if laid {
				Renderable::of(MeshId::QUAD, bumped, Vec3::ONE)
			} else {
				Renderable::new(MeshId::QUAD, Vec3::ONE)
			};
			world.entities.set_renderable(floor, look);

			if thrown {
				covered(&mut world, bumped);
			}

			brightness(
				capture
					.shoot(&mut world)
					.expect("the capture renders")
					.pixel(SQUARE / 2, SQUARE / 2),
			)
		};

		let flat = shoot(&mut capture, false, false);
		let laid = shoot(&mut capture, true, false);
		let thrown = shoot(&mut capture, false, true);

		assert!(
			laid > flat + 60,
			"laid on the floor, the map leans it into the light: {laid} against {flat}"
		);
		assert!(
			thrown.abs_diff(laid) <= 3,
			"and thrown by a decal it leans it the same way by as much: {thrown} against {laid}"
		);
	}

	/// A dark floor filling a square view from straight above, under a light
	/// falling straight down: the middle of the picture is where a smooth
	/// surface shows the light back to the eye.
	///
	/// @return the world, and the floor's own material
	fn floor_under_the_light() -> (World, Material) {
		let mut world = looking_world();
		world.light = Vec3::NEG_Y;
		world.ambient = Vec3::splat(0.1);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let dark = Material::colored(Vec3::splat(0.3));
		let material = world.materials.insert("test/dark", dark);
		let across = view_across(&world);
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		(world, dark)
	}

	#[test]
	fn a_decal_lays_the_roughness_and_the_metal_of_its_material_on_a_surface() {
		// one decal over the whole floor at a time: of the floor's own
		// material, which has to leave it as it was, then a smoother one and a
		// metal one, each of which has to change what the light does there
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let shoot = |capture: &mut Capture, surface: Option<fn(Material) -> Material>| {
			let (mut world, dark) = floor_under_the_light();
			if let Some(surface) = surface {
				let material = world
					.materials
					.insert("test/surface", surface(dark));
				covered(&mut world, material);
			}

			capture
				.shoot(&mut world)
				.expect("the capture renders")
				.pixel(SQUARE / 2, SQUARE / 2)
		};

		let bare = shoot(&mut capture, None);
		let same = shoot(&mut capture, Some(|dark| dark));
		let smooth = shoot(&mut capture, Some(|dark| Material { roughness: 0.35, ..dark }));
		let metal = shoot(&mut capture, Some(|dark| Material { metallic: 1.0, ..dark }));

		assert_eq!(same, bare, "a decal of the floor's own material leaves it as it was");
		assert!(
			brightness(smooth).abs_diff(brightness(bare)) > 30,
			"a smoother one shows the light back where the floor did not: {smooth:?} against \
			 {bare:?}"
		);
		assert!(
			brightness(metal).abs_diff(brightness(bare)) > 30,
			"and a metal one takes the floor's diffuse away: {metal:?} against {bare:?}"
		);
	}

	/// The floor of [`floor_under_the_light`] at a roughness of its own.
	fn floor_of(roughness: f32) -> World {
		let (mut world, dark) = floor_under_the_light();
		world
			.materials
			.insert("test/dark", Material { roughness, ..dark });

		world
	}

	#[test]
	fn a_smoother_floor_shows_the_sun_back_brighter_where_it_mirrors_it() {
		// the middle of the picture is where the floor mirrors the sun into the
		// eye, so it is where the highlight peaks, and a smoother floor gathers
		// the same light into a narrower and taller peak. Exposed at a two
		// hundredth so that the smoothest of the three stays under white, where
		// the arithmetic says 188, 51 and 15 on every channel; the floor the
		// lobe's divisor used to have turned that order upside down, 6, 13, 15.
		//
		// The brightest pixel near the middle rather than the middle itself:
		// the flat normal texel is 128 of 255, a hair past straight out, so an
		// unmapped floor leans a third of a degree and at a tenth the highlight
		// has moved its own half width, a few pixels, off the middle
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let middle = |capture: &mut Capture, roughness: f32| {
			let mut world = floor_of(roughness);
			world.post.exposure = 0.005;

			let picture = capture
				.shoot(&mut world)
				.expect("the capture renders");
			let near = SQUARE / 2 - 8..=SQUARE / 2 + 8;

			near.clone()
				.flat_map(|y| near.clone().map(move |x| (x, y)))
				.map(|(x, y)| picture.pixel(x, y))
				.max_by_key(|pixel| pixel[1])
				.expect("a middle to look at")
		};

		let smooth = middle(&mut capture, 0.1);
		let middling = middle(&mut capture, 0.2);
		let rough = middle(&mut capture, 0.35);

		assert!(
			smooth[1] > middling[1] + 100 && middling[1] > rough[1] + 20,
			"a smoother floor gives back more of the sun where it mirrors it: {smooth:?} at \
			 0.1, {middling:?} at 0.2, {rough:?} at 0.35"
		);
	}

	#[test]
	fn a_floor_of_no_roughness_at_all_is_drawn_as_the_smoothest_the_shader_draws() {
		// nought is under the floor the shader holds a roughness to, so the two
		// pictures are one picture; and a tenth is a picture of its own, which
		// is what says this capture can tell one roughness from the next at all
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let shoot = |capture: &mut Capture, roughness: f32| {
			capture
				.shoot(&mut floor_of(roughness))
				.expect("the capture renders")
		};

		let nought = shoot(&mut capture, 0.0);
		let smoothest = shoot(&mut capture, colby_core::abi::material::MIN_ROUGHNESS);
		let tenth = shoot(&mut capture, 0.1);

		assert!(
			nought.pixels == smoothest.pixels,
			"a surface that says nought is drawn as the smoothest one the shader draws"
		);
		assert!(tenth.pixels != smoothest.pixels, "and one at a tenth is not drawn as either");
	}

	/// A side with a pixel whose middle is the middle of the picture.
	const ODD: u32 = 257;

	#[test]
	fn a_highlight_too_bright_for_the_target_is_written_at_the_ceiling() {
		// a white metal as smooth as the shader draws, a lamp of a hundred one
		// unit above it and the eye straight above both: where the floor mirrors
		// the lamp into the eye it gives back some six million, far past the
		// 65504 a half float holds. Exposed at two to the minus seventeenth the
		// ceiling of two to the fifteenth is a quarter, which is 137 in sRGB.
		// With nothing held, the same pixel reads 187 on a device that
		// saturates a value it cannot store and 255 on one that stores an
		// infinity; with the floor the lobe used to have it read nought
		let Some(mut capture) = capture_of(ODD, ODD) else {
			return;
		};

		// the sun traveling upwards, so the floor is lit by the lamp alone
		let mut world = looking_world();
		world.light = Vec3::Y;
		world.ambient = Vec3::ZERO;
		world.post.exposure = 2.0_f32.powi(-17);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);
		world
			.cvars
			.var(shadow::ENABLED, Value::Bool(false), "off for this picture");

		let mirror = world.materials.insert("test/mirror", Material {
			metallic: 1.0,
			roughness: 0.0,
			..Material::DEFAULT
		});
		let across = view_across(&world);
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, mirror, Vec3::ONE));

		let lamp = world.entities.spawn_at(Transform::at(Vec3::Y));
		assert!(
			world
				.entities
				.set_light(lamp, colby_core::abi::Light::point(Vec3::ONE, 100.0, 5.0))
		);

		let pixel = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(ODD / 2, ODD / 2);

		assert!(
			[pixel[0], pixel[1], pixel[2]]
				.iter()
				.all(|channel| channel.abs_diff(137) <= 1),
			"the highlight is written at the ceiling and no brighter: {pixel:?}"
		);
	}

	#[test]
	fn a_decal_picture_wider_than_the_atlas_takes_whole_is_painted_from_a_smaller_level() {
		// twice as wide as the widest picture the atlas takes whole, so it is
		// packed from its second level on: what lands is the picture, or
		// nothing at all if a level of it went where it does not fit
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let (mut world, _) = floor_below();
		let (wide, tall) = (2048, 8);
		let mut base = Vec::new();
		for _ in 0..wide * tall {
			base.extend_from_slice(&[0xFF, 0x10, 0x10, 0xFF]);
		}
		let picture = world.textures.insert("test/wide", TextureData {
			width: wide,
			height: tall,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: colby_asset::texture::build_chain(wide, tall, base, Texel::Rgba8Srgb)
				.expect("a chain of a picture that size builds"),
		});
		let material = world
			.materials
			.insert("test/wide", Material::textured(picture));
		daubed(&mut world, material, 0);

		let middle = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SQUARE / 2, SQUARE / 2);

		assert_eq!(dominant(middle), 0, "the picture's red lands on the green floor: {middle:?}");
	}

	/// Two texels side by side, red then blue, with no chain under them.
	fn halves() -> TextureData {
		TextureData {
			width: 2,
			height: 1,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![vec![0xFF, 0, 0, 0xFF, 0, 0, 0xFF, 0xFF]],
		}
	}

	/// Renders a quad filling the frame, tiled twice, under one wrap mode.
	///
	/// @param clamp - whether the material holds its textures at their edges
	/// @return the four colors an eighth, three eighths, five eighths and seven
	/// eighths of the way across, or `None` when this machine has no GPU
	fn tiled_twice(clamp: bool) -> Option<Vec<usize>> {
		let mut capture = capture_of(SQUARE, SQUARE)?;

		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let texture = world.textures.insert("test/halves", halves());
		let plain = Material::textured(texture).tiled(2.0);
		let material = world
			.materials
			.insert("test/halves", if clamp { plain.clamped() } else { plain });

		let across = (world.camera.fov_y / 2.0).tan() * HEIGHT * 2.0;
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		Some(
			[1, 3, 5, 7]
				.into_iter()
				.map(|eighth| dominant(image.pixel(SQUARE * eighth / 8, SQUARE / 2)))
				.collect(),
		)
	}

	#[test]
	fn a_material_says_whether_its_textures_tile_or_hold_their_edges() {
		let (Some(repeated), Some(clamped)) = (tiled_twice(false), tiled_twice(true)) else {
			return;
		};

		// tiled twice, so the second copy starts halfway across. Repeating puts
		// the first texel back; clamping has run out of texture by then and
		// holds the last one.
		assert_eq!(repeated, vec![0, 2, 0, 2], "red, blue, red, blue");
		assert_eq!(clamped, vec![0, 2, 2, 2], "red, blue, and then blue forever");
	}

	/// A world with a floor and a light coming in at an angle, so that what a
	/// prop casts lands beside it rather than under it.
	///
	/// A little ambient rather than none: it is what makes "in shadow" a
	/// different reading from "nothing was drawn here", which is the one
	/// mistake a test like this can make.
	fn shadowed_world() -> World {
		let mut world = World::new();
		plainly(&mut world);
		world.clear = rgb(0.0, 0.0, 0.2);
		world.ambient = Vec3::splat(0.12);
		world.light = Vec3::new(1.0, -1.0, 0.0).normalize();

		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(120.0, 1.0, 120.0),
		});
		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::QUAD, Vec3::ONE));

		world
	}

	/// Where a point in the world lands in the picture.
	///
	/// Through the same matrix the renderer uses, so a test never has to
	/// re-derive a projection and then be wrong about it. Call it after
	/// shooting, because that is when `aspect` is the capture's.
	///
	/// @param world - whose camera and aspect to project through
	/// @param point - where in the world
	/// @param size - the capture's width and height
	fn on_screen(world: &World, point: Vec3, size: (u32, u32)) -> (u32, u32) {
		let ndc = world
			.render_camera()
			.view_projection(world.aspect)
			.project_point3(point);

		let across = f32::from(u16::try_from(size.0).unwrap_or(u16::MAX));
		let down = f32::from(u16::try_from(size.1).unwrap_or(u16::MAX));

		(
			pixel(ndc.x.mul_add(0.5, 0.5) * across, across),
			pixel(ndc.y.mul_add(-0.5, 0.5) * down, down),
		)
	}

	/// One axis of a projected point as a pixel inside the picture.
	///
	/// Clamped rather than trusted: a point behind the camera projects to
	/// something that is not a pixel at all, and a test that asks for one
	/// should be told about it by the color it reads rather than by an
	/// arithmetic panic in the middle of a render.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "held inside zero and the picture's own size on the line above the cast, which 		          is the check the cast itself would not do"
	)]
	fn pixel(value: f32, limit: f32) -> u32 { value.clamp(0.0, limit - 1.0).round() as u32 }

	/// Where a straight ray of light from a point lands on the floor at `y =
	/// 0`.
	fn beneath(world: &World, caster: Vec3) -> Vec3 {
		let direction = world.light.normalize();

		caster - direction * (caster.y / direction.y)
	}

	/// A device on one named API, or `None` when this build has not got it.
	///
	/// Its own rather than the shared one, and the shared one is not used
	/// here at all: what this test is about is two backends, and the shared
	/// device is whichever of them wgpu ranked first. Asking for each by name
	/// is the only way to know which two were compared.
	fn on(api: Backends) -> Option<&'static Gpu> {
		static SECOND: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();

		// **The shared device for whichever API it is already on, and one more
		// held for the life of the process for the other.** Every other test in
		// this binary draws on [`crate::gpu::shared`], and the reason is
		// measured rather than tidy: a suite that opens a device per test has a
		// dozen alive at once and the driver falls over - an access violation
		// with no panic and no failing test, the harness simply stopping
		// mid-list. This test is the one that cannot use the shared device for
		// *both* of its devices, because its whole point is two different APIs.
		// So it uses it for one of them and opens the other once, which leaves
		// the process with two devices rather than four.
		let shared = crate::gpu::shared()?;
		if Backends::from(shared.adapter().get_info().backend) == api {
			return Some(shared);
		}

		SECOND
			.get_or_init(|| match Gpu::open(api, None) {
				| Ok(gpu) => gpu,
				| Err(error) => panic!("opening a {api:?} device failed: {error}"),
			})
			.as_ref()
	}

	#[test]
	fn the_two_graphics_apis_this_build_has_draw_the_same_picture() {
		// **the whole of what says a second backend is covered rather than
		// claimed.** `r.backend` takes more than one word, and until this
		// nothing in the gate ever drew with the second one: the default is
		// whichever wgpu ranks first, and on every machine here that is
		// Vulkan. Enabling an API and never rendering through it is a feature
		// list, not a test.
		//
		// **Within a level or two and not byte for byte**, which was measured
		// rather than guessed: on 2026-09-08 the same 1280x720 frame of a
		// landscape with terrain, shadows and the whole post chain came out
		// with 13.83% of its pixels differing between the two and **not one
		// channel differing by more than one**. Two rasterizers rounding the
		// same arithmetic differently is what that is; anything larger is a
		// backend drawing something else. On this test's own smaller scene
		// the two agree byte for byte, which is why the bound is written
		// from the landscape rather than from what passes here.
		let (Some(vulkan), Some(dx12)) = (on(Backends::VULKAN), on(Backends::DX12)) else {
			return;
		};

		// and the two really are two: a bound nothing approaches would pass
		// just as happily on one device compared with itself, which is the
		// shape this test would fail at silently.
		assert_ne!(
			vulkan.adapter().get_info().backend,
			dx12.adapter().get_info().backend,
			"each device is on the API it was asked for"
		);

		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let cube = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 3.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.2, 0.1)));

		let shots = [&vulkan, &dx12].map(|gpu| {
			let mut capture =
				Capture::new(gpu, SIZE.0, SIZE.1).expect("a capture on each of them builds");

			capture
				.shoot(&mut world)
				.expect("each of them renders the frame")
		});
		let (first, second) = (&shots[0], &shots[1]);

		// the half that says the two are pictures of something. Two backends
		// that both drew nothing would agree perfectly, and that is exactly
		// the failure this test would otherwise pass.
		let middle = first.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(dominant(middle), 0, "the cube is in the middle of the picture: {middle:?}");
		assert!(middle[0] > 60, "and it is lit rather than black: {middle:?}");

		let worst = (0..SIZE.1)
			.flat_map(|y| (0..SIZE.0).map(move |x| (x, y)))
			.map(|(x, y)| distance(first.pixel(x, y), second.pixel(x, y)))
			.max()
			.unwrap_or(0);

		assert!(
			worst <= 2,
			"the two graphics APIs disagree by {worst} levels, which is more than rounding"
		);
	}

	#[test]
	fn a_prop_above_a_floor_casts_a_shadow_onto_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let at = Vec3::new(0.0, 3.0, 0.0);
		let cube = world.entities.spawn_at(Transform {
			position: at,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(1.4),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		let shaded = on_screen(&world, beneath(&world, at), SIZE);
		let beside = on_screen(&world, beneath(&world, at) + Vec3::new(0.0, 0.0, 3.5), SIZE);

		let dark = brightness(image.pixel(shaded.0, shaded.1));
		let lit = brightness(image.pixel(beside.0, beside.1));

		assert!(
			dark * 2 < lit,
			"the floor under the prop at {shaded:?} reads {dark} against {lit} beside it"
		);
		assert!(
			dark > 20,
			"and it is in shadow rather than missing: what is left there is the ambient, got \
			 {dark}"
		);

		// the same frame with the console variable off. Written into the world
		// by hand, which is what a pixel test can do and `--shot` cannot.
		world
			.cvars
			.var(shadow::ENABLED, Value::Bool(false), "off for this frame");

		let unshadowed = capture
			.shoot(&mut world)
			.expect("the second capture renders");
		let without = brightness(unshadowed.pixel(shaded.0, shaded.1));

		assert!(
			without * 10 > lit * 9,
			"with the cascades off that spot is as lit as the floor beside it: {without} \
			 against {lit}"
		);
	}

	/// A world with a cube in the middle of the view and one in each of three
	/// places the view does not reach: behind the eye, off to the left, and
	/// past the far plane.
	///
	/// The one behind the eye also stands between the light and the one in the
	/// middle, so its shadow is on the face the camera sees: the picture holds
	/// something only a caster outside the view can put there.
	fn surrounded() -> World {
		let mut world = looking_world();
		world.ambient = Vec3::splat(0.3);

		for (at, color) in [
			(Vec3::ZERO, rgb(0.9, 0.1, 0.1)),
			(Vec3::new(0.0, 0.0, 12.0), rgb(0.1, 0.9, 0.1)),
			(Vec3::new(-30.0, 0.0, 0.0), rgb(0.1, 0.1, 0.9)),
			(Vec3::new(0.0, 0.0, -300.0), rgb(0.9, 0.9, 0.1)),
		] {
			let id = world.entities.spawn_at(Transform::at(at));

			world
				.entities
				.set_renderable(id, Renderable::new(MeshId::CUBE, color));
		}

		world
	}

	#[test]
	fn what_the_camera_cannot_see_is_left_out_and_the_picture_does_not_change() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = surrounded();
		let culled = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let kept = capture.scene_mut().drawn();

		world
			.cvars
			.var(cull::ENABLED, Value::Bool(false), "off for this frame");

		let whole = capture
			.shoot(&mut world)
			.expect("the second capture renders");
		let everything = capture.scene_mut().drawn();

		assert_eq!(kept.meshes, 4, "four entities have a mesh: {kept:?}");
		assert_eq!(kept.seen, 1, "and only the one in front of the camera is drawn: {kept:?}");
		assert_eq!(everything.seen, 4, "with the test off all four are: {everything:?}");
		assert_eq!(
			everything.cast,
			shadow::CASCADES * 4,
			"and every cascade draws every one of them: {everything:?}"
		);
		assert!(
			kept.cast < everything.cast,
			"while culling the cascades draw fewer: {kept:?} against {everything:?}"
		);
		assert_eq!(
			dominant(culled.pixel(SIZE.0 / 2, SIZE.1 / 2)),
			0,
			"the one that is drawn is in the picture, so the pictures are of something"
		);
		assert!(
			culled.pixels == whole.pixels,
			"and the picture with three cubes left out is the picture with none left out"
		);
	}

	#[test]
	fn a_caster_outside_the_view_still_throws_its_shadow_into_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		// twelve units off towards the side the light comes from and six up:
		// out of the picture altogether, and its shadow lands six units in
		let at = Vec3::new(-12.0, 6.0, 0.0);
		let cube = world.entities.spawn_at(Transform {
			position: at,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(1.4),
		});
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, Vec3::ONE));

		let culled = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let drawn = capture.scene_mut().drawn();

		let shaded = on_screen(&world, beneath(&world, at), SIZE);
		let beside = on_screen(&world, beneath(&world, at) + Vec3::new(0.0, 0.0, 3.5), SIZE);
		let dark = brightness(culled.pixel(shaded.0, shaded.1));
		let lit = brightness(culled.pixel(beside.0, beside.1));

		assert_eq!(drawn.seen, 1, "the picture draws the floor and not the cube: {drawn:?}");
		assert!(
			dark * 2 < lit,
			"and the cube's shadow is on the floor at {shaded:?} all the same: {dark} against \
			 {lit} beside it"
		);

		world
			.cvars
			.var(cull::ENABLED, Value::Bool(false), "off for this frame");

		let whole = capture
			.shoot(&mut world)
			.expect("the second capture renders");

		assert!(
			culled.pixels == whole.pixels,
			"and it is the same shadow the frame with nothing left out throws"
		);
	}

	/// A cube a little bigger than the unit one, standing somewhere in a color.
	fn cube_at(world: &mut World, at: Vec3, color: Vec3) -> EntityId {
		let id = world.entities.spawn_at(Transform {
			position: at,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(1.4),
		});
		world
			.entities
			.set_renderable(id, Renderable::new(MeshId::CUBE, color));

		id
	}

	#[test]
	fn hiding_a_parent_takes_what_hangs_off_it_out_of_the_picture_and_the_shadows() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the caster from the test above, out of the view with its shadow in
		// it, and a red cube in the view, both hanging off one entity that
		// draws nothing; and a green cube standing on its own, which is the
		// part of the picture a hide has to leave alone
		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let group = world.entities.spawn();
		let outside = Vec3::new(-12.0, 6.0, 0.0);
		let inside = Vec3::new(3.0, 1.0, -2.0);
		let alone = Vec3::new(0.0, 1.0, 3.0);
		let caster = cube_at(&mut world, outside, Vec3::ONE);
		let red = cube_at(&mut world, inside, rgb(0.9, 0.1, 0.1));
		cube_at(&mut world, alone, rgb(0.1, 0.9, 0.1));
		assert!(world.entities.set_parent(caster, group), "the caster hangs off the group");
		assert!(world.entities.set_parent(red, group), "and so does the red cube");

		let before = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let all = capture.scene_mut().drawn();
		let shaded = on_screen(&world, beneath(&world, outside), SIZE);
		let beside = on_screen(&world, beneath(&world, outside) + Vec3::new(0.0, 0.0, 3.5), SIZE);
		let cube = on_screen(&world, inside, SIZE);
		let lone = on_screen(&world, alone, SIZE);
		let lit = brightness(before.pixel(beside.0, beside.1));

		assert!(
			brightness(before.pixel(shaded.0, shaded.1)) * 2 < lit,
			"the caster's shadow is in the picture to begin with"
		);
		assert_eq!(dominant(before.pixel(cube.0, cube.1)), 0, "and so is the red cube");

		assert!(world.entities.set_hidden(group, true), "the handle resolves");

		let after = capture
			.shoot(&mut world)
			.expect("the second capture renders");
		let left = capture.scene_mut().drawn();
		let where_red_was = after.pixel(cube.0, cube.1);

		assert_eq!(left.hidden, 2, "both of the group's cubes were left out: {left:?}");
		assert_eq!(
			left.meshes + left.hidden,
			all.meshes,
			"and they are the whole difference: {left:?} against {all:?}"
		);
		assert_eq!(left.seen + 1, all.seen, "the picture lost the red cube: {left:?}");
		assert!(left.cast < all.cast, "and the cascades both: {left:?} against {all:?}");
		assert!(
			brightness(after.pixel(shaded.0, shaded.1)) * 10 > lit * 9,
			"the shadow went with its caster"
		);
		assert!(
			u32::from(where_red_was[0]) < u32::from(where_red_was[1]) + 30,
			"the red cube is not drawn, got {where_red_was:?}"
		);
		assert_eq!(
			after.pixel(lone.0, lone.1),
			before.pixel(lone.0, lone.1),
			"and the green one standing on its own is untouched"
		);

		assert!(world.entities.set_hidden(group, false));

		let again = capture
			.shoot(&mut world)
			.expect("the third capture renders");

		assert!(
			again.pixels == before.pixels,
			"shown again, the picture is the first one to the byte"
		);
		assert_eq!(capture.scene_mut().drawn(), all, "and so are the counts");
	}

	#[test]
	fn a_lamp_hung_off_something_hidden_lights_nothing_until_it_is_shown() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the sun traveling a little upwards, so the floor facing up is lit
		// by the lamp alone and the spot under the lamp is the lamp's
		let mut world = shadowed_world();
		world.light = Vec3::new(1.0, 0.3, 0.0).normalize();
		world.ambient = Vec3::splat(0.05);
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;
		world
			.cvars
			.var(shadow::ENABLED, Value::Bool(false), "off for these pictures");

		let group = world.entities.spawn();
		let lamp = world.entities.spawn_at(Transform::at(Vec3::Y));
		assert!(
			world
				.entities
				.set_light(lamp, colby_core::abi::Light::point(Vec3::ONE, 4.0, 6.0))
		);
		assert!(world.entities.set_parent(lamp, group));

		let lit = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let under = on_screen(&world, Vec3::ZERO, SIZE);

		assert_eq!(capture.scene_mut().drawn().lamps, 1, "the lamp is carried");

		assert!(world.entities.set_hidden(group, true));

		let dark = capture
			.shoot(&mut world)
			.expect("the second capture renders");

		assert_eq!(
			capture.scene_mut().drawn().lamps,
			0,
			"and hidden by what it hangs off, it is not"
		);
		assert!(
			brightness(dark.pixel(under.0, under.1)) * 2
				< brightness(lit.pixel(under.0, under.1)),
			"so the spot under it goes dark"
		);

		assert!(world.entities.set_hidden(group, false));

		let again = capture
			.shoot(&mut world)
			.expect("the third capture renders");

		assert!(again.pixels == lit.pixels, "and shown again it lights the same picture");
	}

	#[test]
	fn a_cube_far_down_the_view_is_cast_into_some_cascades_and_not_all() {
		let Some(mut capture) = capture() else {
			return;
		};

		// forty units down the view and fifteen to the right: in the picture,
		// and inside the last cascade's box, but beyond the far side of the
		// nearer three, whose boxes stop well short of it. **The one thing a
		// picture cannot say**: a cube drawn into a cascade that clips it away
		// changes no pixel, so what a list that forgot which cascade it is for
		// would cost shows up here and nowhere else.
		let mut world = looking_world();
		let cube = world
			.entities
			.spawn_at(Transform::at(Vec3::new(15.0, 0.0, -35.0)));
		world
			.entities
			.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.1, 0.1)));

		capture
			.shoot(&mut world)
			.expect("the capture renders");
		let drawn = capture.scene_mut().drawn();

		assert_eq!(drawn.seen, 1, "it is in the picture: {drawn:?}");
		assert!(
			(1..shadow::CASCADES).contains(&drawn.cast),
			"and in some of the cascades' lists but not in every one: {drawn:?}"
		);
	}

	/// The bar from [`armed`], standing somewhere of the caller's choosing.
	fn armed_at(at: Vec3) -> (World, PoseId) {
		let mut world = looking_world();
		let mesh = world.meshes.insert(BAR, bar([0, 255, 0, 0]));
		let skeleton = world.skeletons.insert("rig", elbow());
		let pose = world
			.poses
			.spawn(Pose::resting(skeleton, world.skeletons.bones(skeleton)));
		let id = world.entities.spawn_at(Transform::at(at));

		world
			.entities
			.set_renderable(id, Renderable::new(mesh, rgb(0.9, 0.7, 0.2)).posed(pose));

		(world, pose)
	}

	/// Carries a whole pose along `x` by moving its root bone, which is what a
	/// ragdoll does to a character: the entity stays and the bones go.
	fn carried(world: &mut World, pose: PoseId, along: f32) {
		world.advance();
		world
			.poses
			.get_mut(pose)
			.expect("the pose is there")
			.set(0, Transform::at(Vec3::X * along));
		world.poses.snap_all();
		world.settle();
	}

	#[test]
	fn a_mesh_its_bones_carry_into_the_view_is_drawn_though_it_rests_outside_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// eleven units off to the left, where the camera cannot see the bar
		// in the shape it was modeled in
		let (mut world, pose) = armed_at(Vec3::new(-12.0, -0.5, 0.0));
		let resting = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(capture.scene_mut().drawn().seen, 0, "at rest it is out of the picture");
		assert!(top_row(&resting, resting.pixel(1, 1)).is_none(), "and nothing is drawn");

		// and its root carries it eleven back in
		carried(&mut world, pose, 11.0);

		let culled = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(capture.scene_mut().drawn().seen, 1, "carried in, it is drawn");
		assert!(
			top_row(&culled, culled.pixel(1, 1)).is_some(),
			"and it is in the picture, where its bones put it"
		);

		world
			.cvars
			.var(cull::ENABLED, Value::Bool(false), "off for this frame");

		let whole = capture
			.shoot(&mut world)
			.expect("the second capture renders");

		assert!(culled.pixels == whole.pixels, "the same picture as with nothing left out");
	}

	#[test]
	fn a_mesh_its_bones_carry_out_of_the_view_is_left_out_though_it_rests_inside_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the other way round: in the middle of the picture at rest, and
		// carried twenty units off to the right by its root
		let (mut world, pose) = armed_at(Vec3::new(-1.0, -0.5, 0.0));

		capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(capture.scene_mut().drawn().seen, 1, "at rest it is drawn");

		carried(&mut world, pose, 20.0);

		let culled = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(
			capture.scene_mut().drawn().seen,
			0,
			"carried away it is not, though where it rests is in the middle of the view"
		);
		assert!(top_row(&culled, culled.pixel(1, 1)).is_none(), "and nothing is drawn");
	}

	/// How far above the floor the holed caster hangs, and how wide it is.
	///
	/// Three and three: the light comes in at forty-five degrees, so what the
	/// caster throws lands exactly its own width to `+x` of it - beside itself
	/// rather than under itself, where an overhead camera can see it and where
	/// the caster's own holes are not in the way.
	const CASTER: (f32, f32) = (3.0, 3.0);

	/// How many times the caster's picture repeats across it.
	///
	/// Two rather than one, and that is the difference between a test and a
	/// decoration: at one tile every sample below lands on the same texel
	/// whether or not the cascade pass multiplies the coordinate by the
	/// material's scale, so a pass that dropped the multiply entirely would
	/// pass. At two, the second sample is a hole only if the multiply happened.
	const TILES: f32 = 2.0;

	/// One frame of a quad wearing [`holed`] hung over the floor, in a mode.
	///
	/// A quad rather than a cube, so that the picture's four texels become four
	/// squares of shadow on the floor and two of them are what the mask is
	/// supposed to take away.
	///
	/// @param blend - how the caster's material reads the picture's alpha
	/// @return the frame and the world it was shot from, or `None` with no GPU
	fn cast_by(blend: Blend) -> Option<(Image, World)> {
		let mut capture = capture()?;
		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let texture = world.textures.insert("test/holed", holed());
		let material = world.materials.insert("test/holed", Material {
			blend,
			..Material::textured(texture).tiled(TILES)
		});

		let caster = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, CASTER.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(CASTER.1, 1.0, CASTER.1),
		});
		world
			.entities
			.set_renderable(caster, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		Some((image, world))
	}

	#[test]
	fn a_cutout_casts_a_shadow_with_the_holes_in_it() {
		let (Some((solid, world)), Some((cut, _))) =
			(cast_by(Blend::Opaque), cast_by(Blend::Mask))
		else {
			return;
		};

		// the middles of two neighboring cells of the tiled picture, in the
		// world. The first is opaque and the second is a hole; they are
		// neighbors at the same height in the same row, so nothing about them
		// differs but the alpha the cascade pass reads.
		let cell = CASTER.1 / (2.0 * TILES);
		let middle = |column: f32| (column + 0.5).mul_add(cell, -CASTER.1 / 2.0);
		let kept = Vec3::new(middle(0.0), CASTER.0, middle(0.0));
		let hole = Vec3::new(middle(1.0), CASTER.0, middle(0.0));

		let under_kept = on_screen(&world, beneath(&world, kept), SIZE);
		let under_hole = on_screen(&world, beneath(&world, hole), SIZE);
		let beside = on_screen(&world, beneath(&world, kept) + Vec3::new(0.0, 0.0, 4.0), SIZE);

		assert_ne!(under_kept, under_hole, "the two samples are different pixels");

		let lit = brightness(solid.pixel(beside.0, beside.1));

		// the control, and the half of this test that says the shadow reaches
		// the second point at all: drawn solid, both texels shade the floor.
		for (name, (x, y)) in [("kept", under_kept), ("hole", under_hole)] {
			let dark = brightness(solid.pixel(x, y));

			assert!(
				dark * 2 < lit,
				"drawn solid, the {name} texel shades the floor at ({x}, {y}): {dark} against \
				 {lit} beside it"
			);
		}

		let still_dark = brightness(cut.pixel(under_kept.0, under_kept.1));
		let now_lit = brightness(cut.pixel(under_hole.0, under_hole.1));

		assert!(
			still_dark * 2 < lit,
			"masked, the texel that is not a hole goes on shading the floor: {still_dark} \
			 against {lit}"
		);
		assert!(
			now_lit * 10 > lit * 8,
			"and the one that is a hole lets the light through: {now_lit} against {lit} beside \
			 it"
		);
		// what makes the pair a measurement rather than two coincidences: a
		// build whose cascades never reached the masked pipeline would shade
		// both points in both modes, and one that discarded everything would
		// shade neither.
		assert!(
			now_lit > still_dark * 2,
			"so inside one frame the hole and the solid texel really do differ: {now_lit} \
			 against {still_dark}"
		);
	}

	/// The built-in quad with every vertex painted one color.
	///
	/// @param color - linear RGBA, as the paint holds it
	fn painted_quad(color: Vec4) -> MeshData {
		let mut quad = mesh::quad();
		quad.paint = vec![PaintVertex::new(color, Vec2::ZERO); quad.vertices.len()];

		quad
	}

	/// A flat square in the xz plane, facing up, cut into `side` by `side`
	/// cells: the built-in quad with more vertices than it needs.
	///
	/// @param side - how many cells along each edge
	fn grid(side: u16) -> MeshData {
		let cells = f32::from(side);
		let mut data = MeshData::default();

		for row in 0..=side {
			for column in 0..=side {
				let (u, v) = (f32::from(column) / cells, f32::from(row) / cells);

				data.vertices
					.push(colby_core::abi::MeshVertex::new(
						Vec3::new(u - 0.5, 0.0, v - 0.5),
						Vec3::Y,
						Vec2::new(u, v),
					));
			}
		}

		let wide = u32::from(side) + 1;

		for row in 0..u32::from(side) {
			for column in 0..u32::from(side) {
				let corner = row * wide + column;

				data.indices.extend([
					corner,
					corner + wide,
					corner + 1,
					corner + 1,
					corner + wide,
					corner + wide + 1,
				]);
			}
		}

		mesh::tangents(&mut data);

		data
	}

	/// One overhead frame of whatever a caller stands in it, lit from
	/// everywhere and by nothing else, with the frame's own size in the world
	/// handed over so a floor can fill it.
	///
	/// The arrangement of [`shot_of`], opened up: the camera straight above,
	/// the light traveling sideways so that a surface facing up is lit by the
	/// ambient term alone, and a blue clear under everything.
	///
	/// @param dress - what stands in the world, given the world and how wide
	/// the frame is at the floor
	/// @return the frame, or `None` on a machine with no GPU
	fn overhead<F>(dress: F) -> Option<Image>
	where
		F: FnOnce(&mut World, f32),
	{
		let mut capture = capture_of(SQUARE, SQUARE)?;
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		let across = (world.camera.fov_y / 2.0).tan() * HEIGHT * 2.0;
		dress(&mut world, across);

		Some(
			capture
				.shoot(&mut world)
				.expect("the capture renders"),
		)
	}

	/// Stands a mesh in a material at a height, as wide as the frame.
	fn stand(world: &mut World, across: f32, height: f32, mesh: MeshId, material: MaterialId) {
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, height, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(mesh, material, Vec3::ONE));
	}

	#[test]
	fn a_vertex_painted_a_color_is_drawn_as_the_same_color_on_its_material_is() {
		let color = Vec4::new(0.25, 0.5, 0.75, 1.0);
		// the color as the paint holds it, sixteen bits a channel, which is what
		// the material has to be given for the two to be one color
		let held = PaintVertex::new(color, Vec2::ZERO)
			.color()
			.truncate();

		let (Some(by_paint), Some(by_material), Some(white)) = (
			overhead(|world, across| {
				let mesh = world
					.meshes
					.insert("test/painted", painted_quad(color));

				stand(world, across, 0.0, mesh, MaterialId::DEFAULT);
			}),
			overhead(|world, across| {
				let material = world
					.materials
					.insert("test/colored", Material::colored(held));

				stand(world, across, 0.0, MeshId::QUAD, material);
			}),
			overhead(|world, across| {
				stand(world, across, 0.0, MeshId::QUAD, MaterialId::DEFAULT);
			}),
		) else {
			return;
		};

		assert!(
			by_paint.pixels == by_material.pixels,
			"a floor painted a color and a floor made of it are one picture, to the byte"
		);
		// what makes that a measurement: a paint the shader never read would
		// hand back the white floor, and the relation above would then fail
		// only because the material half moved
		assert!(
			distance(by_paint.pixel(SQUARE / 2, SQUARE / 2), white.pixel(SQUARE / 2, SQUARE / 2))
				> 40,
			"and neither is the white floor: {:?} against {:?}",
			by_paint.pixel(SQUARE / 2, SQUARE / 2),
			white.pixel(SQUARE / 2, SQUARE / 2)
		);
	}

	#[test]
	fn a_cutout_painted_clear_leaves_nothing_of_itself_and_a_solid_one_ignores_it() {
		let clear = Vec4::new(1.0, 1.0, 1.0, 0.0);
		let dressed = |blend: Blend| {
			overhead(move |world, across| {
				let mesh = world
					.meshes
					.insert("test/clear", painted_quad(clear));
				let material = world
					.materials
					.insert("test/mode", Material { blend, ..Material::DEFAULT });

				stand(world, across, 0.0, mesh, material);
			})
		};

		let (Some(cut), Some(solid), Some(nothing), Some(white)) = (
			dressed(Blend::Mask),
			dressed(Blend::Opaque),
			overhead(|_, _| ()),
			overhead(|world, across| {
				stand(world, across, 0.0, MeshId::QUAD, MaterialId::DEFAULT);
			}),
		) else {
			return;
		};

		assert!(
			cut.pixels == nothing.pixels,
			"a cutout painted clear at every vertex is a hole all the way across"
		);
		assert!(
			solid.pixels == white.pixels,
			"and a solid surface reads no alpha, painted or not, which leaves the white floor"
		);
		assert!(cut.pixels != solid.pixels, "so the two modes really do differ here");
	}

	/// One overhead frame of a red floor, with a pane of blue glass over it
	/// painted to some alpha, or with none.
	///
	/// @param alpha - how much of the picture the pane's paint leaves, or
	/// `None` for no pane at all
	fn pane_over_red(alpha: Option<f32>) -> Option<Image> {
		overhead(move |world, across| {
			let red = world
				.materials
				.insert("test/red", Material::colored(Vec3::new(0.8, 0.1, 0.1)));
			stand(world, across, 0.0, MeshId::QUAD, red);

			let Some(alpha) = alpha else {
				return;
			};

			let mesh = world
				.meshes
				.insert("test/pane", painted_quad(Vec4::new(1.0, 1.0, 1.0, alpha)));
			let glass = world.materials.insert(
				"test/glass",
				Material::colored(Vec3::new(0.1, 0.1, 0.9)).translucent(1.0),
			);

			stand(world, across, 1.0, mesh, glass);
		})
	}

	#[test]
	fn glass_painted_clear_lets_through_everything_behind_it() {
		let (Some(bare), Some(clear), Some(seen)) =
			(pane_over_red(None), pane_over_red(Some(0.0)), pane_over_red(Some(1.0)))
		else {
			return;
		};

		assert!(
			clear.pixels == bare.pixels,
			"a pane painted clear is not there: the floor behind it, to the byte"
		);
		assert!(
			distance(seen.pixel(SQUARE / 2, SQUARE / 2), bare.pixel(SQUARE / 2, SQUARE / 2)) > 40,
			"and one painted whole is: {:?} against {:?}",
			seen.pixel(SQUARE / 2, SQUARE / 2),
			bare.pixel(SQUARE / 2, SQUARE / 2)
		);
	}

	/// One frame of a white quad painted to some alpha hung over the floor, in
	/// a mode: [`cast_by`] with the holes in the paint rather than the picture.
	fn cast_painted(blend: Blend, alpha: f32) -> Option<(Image, World)> {
		let mut capture = capture()?;
		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		let mesh = world
			.meshes
			.insert("test/pane", painted_quad(Vec4::new(1.0, 1.0, 1.0, alpha)));
		let material = world
			.materials
			.insert("test/pane", Material { blend, ..Material::DEFAULT });
		let caster = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, CASTER.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(CASTER.1, 1.0, CASTER.1),
		});
		world
			.entities
			.set_renderable(caster, Renderable::of(mesh, material, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		Some((image, world))
	}

	#[test]
	fn a_cutout_painted_clear_throws_no_shadow_and_a_solid_one_painted_clear_still_does() {
		let (Some((cut, world)), Some((solid, _)), Some((kept, _))) = (
			cast_painted(Blend::Mask, 0.0),
			cast_painted(Blend::Opaque, 0.0),
			cast_painted(Blend::Mask, 1.0),
		) else {
			return;
		};

		// under the middle of the caster, and far enough to its side that no
		// shadow reaches
		let under = on_screen(&world, beneath(&world, Vec3::new(0.0, CASTER.0, 0.0)), SIZE);
		let beside = on_screen(
			&world,
			beneath(&world, Vec3::new(0.0, CASTER.0, 0.0)) + Vec3::new(0.0, 0.0, 4.0),
			SIZE,
		);
		let lit = brightness(solid.pixel(beside.0, beside.1));

		assert!(
			brightness(solid.pixel(under.0, under.1)) * 2 < lit,
			"a solid pane shades the floor whatever it was painted"
		);
		assert!(
			brightness(kept.pixel(under.0, under.1)) * 2 < lit,
			"and so does a cutout painted whole"
		);
		assert!(
			brightness(cut.pixel(under.0, under.1)) * 10 > lit * 8,
			"but a cutout painted clear lets the light through, because the cascade pass reads 			 the paint's alpha where the scene does: {} against {lit}",
			brightness(cut.pixel(under.0, under.1))
		);
	}

	#[test]
	fn a_mesh_longer_than_the_plain_buffer_still_reads_white_at_every_vertex() {
		// seventy cells a side is 5,041 vertices, past the 1,024 the plain
		// buffer starts with: a buffer that did not grow would hand the far end
		// of the mesh whatever lies past its own end
		let side = 70;
		let plainly_white = |painted: bool| {
			overhead(move |world, across| {
				let mut mesh = grid(side);
				// a block of plain entries as long as the mesh, or none at all
				mesh.paint = vec![PaintVertex::PLAIN; mesh.vertices.len() * usize::from(painted)];

				let mesh = world.meshes.insert("test/grid", mesh);

				stand(world, across, 0.0, mesh, MaterialId::DEFAULT);
			})
		};

		let (Some(read_plain), Some(painted_plain)) = (plainly_white(false), plainly_white(true))
		else {
			return;
		};

		assert!(
			read_plain.pixels == painted_plain.pixels,
			"a long mesh nobody painted draws as one painted white at every vertex, to the byte"
		);

		let Some(gpu) = crate::gpu::shared() else {
			return;
		};
		let mut capture = Capture::new(gpu, 8, 8).expect("a small capture builds");
		let mut world = World::new();
		let mesh = world.meshes.insert("test/grid", grid(side));
		let floor = world.entities.spawn();
		world
			.entities
			.set_renderable(floor, Renderable::new(mesh, Vec3::ONE));

		capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(
			capture.scene_mut().plain_vertices() >= 71 * 71,
			"and the buffer standing in for its paint grew to hold it: {}",
			capture.scene_mut().plain_vertices()
		);
	}

	#[test]
	fn both_shaders_cut_their_holes_out_at_the_same_place() {
		// a module cannot include another one and neither can read a constant
		// out of Rust, so the number is written three times and this is what
		// stops the three drifting. A fence whose shadow had different holes in
		// it than the fence has is the exact bug that would follow, and it is
		// invisible on anything but a lit scene; the importer reading the third
		// copy would quietly warn about files that are in fact exact.
		let declaration = "const MASK_CUTOFF: f32 = ";
		let cutoff = |source: &str, name: &str| {
			source
				.split_once(declaration)
				.and_then(|(_, rest)| rest.split_once(';'))
				.map(|(value, _)| value.trim().to_owned())
				.unwrap_or_else(|| panic!("{name} declares no {declaration}"))
		};

		let scene = cutoff(include_str!("shader.wgsl"), "shader.wgsl");

		assert_eq!(
			scene,
			cutoff(include_str!("shadow.wgsl"), "shadow.wgsl"),
			"the scene and the cascades have to agree on which texels are holes"
		);
		assert_eq!(
			scene.parse::<f32>().ok(),
			Some(colby_core::abi::material::MASK_CUTOFF),
			"and so does the one the importer measures a file's own cutoff against"
		);
	}

	/// How much of a pane there is, where a test does not care about the exact
	/// number.
	///
	/// A half rather than anything nearer an end: at nothing the pane is not
	/// there and at one it is a wall, and either of those is a picture a build
	/// that never blended anything would also produce.
	const HALF: f32 = 0.5;

	/// Puts a horizontal pane over the origin and hands back its handle.
	///
	/// Horizontal and lit by the ambient alone, so what comes out is the color
	/// it was given rather than something a light angle decided.
	///
	/// @param world - where to put it
	/// @param name - what to register its material as, which also orders two of
	/// them: the registry hands out slots in the order it is asked
	/// @param color - its base color
	/// @param height - how far above the origin it hangs
	/// @param blend - how its alpha is read
	fn pane(world: &mut World, name: &str, color: Vec3, height: f32, blend: Blend) -> MaterialId {
		let material = world.materials.insert(name, Material {
			blend,
			opacity: HALF,
			..Material::colored(color)
		});

		let entity = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, height, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(3.0, 1.0, 3.0),
		});
		world
			.entities
			.set_renderable(entity, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

		material
	}

	/// A world lit by nothing but a full ambient, seen from above.
	///
	/// The panes below are horizontal and the light travels sideways, so every
	/// surface here is lit by the ambient term alone and comes out the color it
	/// was given. That is what makes a channel comparison mean the compositing
	/// rather than the shading.
	fn overhead_world() -> World {
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);
		world.camera.position = Vec3::new(0.0, 9.0, 0.01);
		world.camera.target = Vec3::ZERO;

		world
	}

	#[test]
	fn a_blended_surface_lets_what_is_behind_it_through() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut shot = |blend, cube_under: bool| {
			let mut world = overhead_world();

			if cube_under {
				let cube = world.entities.spawn_at(Transform {
					position: Vec3::ZERO,
					rotation: Quat::IDENTITY,
					scale: Vec3::splat(2.0),
				});
				world
					.entities
					.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.05, 0.05)));
			}

			pane(&mut world, "test/pane", rgb(0.05, 0.9, 0.05), 2.0, blend);

			capture
				.shoot(&mut world)
				.expect("the capture renders")
				.pixel(SIZE.0 / 2, SIZE.1 / 2)
		};

		// the pane over nothing at all, which is the only honest reading of
		// "the cube did not show through": the pane's own color has some red in
		// it, and guessing a number for how much would be a test of the sRGB
		// curve rather than of the compositing.
		let bare = shot(Blend::Opaque, false);
		let hidden = shot(Blend::Opaque, true);
		let through = shot(Blend::Alpha, true);

		assert_eq!(dominant(bare), 1, "the pane is the green it was given: {bare:?}");
		assert_eq!(
			bare[3], 255,
			"and the frame it left behind is opaque: a surface this pipeline draws is solid 			 \
			 whatever opacity the material carries, which is {HALF}"
		);
		assert!(
			distance(hidden, bare) <= 1,
			"drawn solid it hides the cube completely: {hidden:?} against {bare:?} with nothing 			 under it"
		);
		assert!(
			u32::from(through[0]) > u32::from(hidden[0]) + 40,
			"drawn blended the cube shows through it: {through:?} against {hidden:?}"
		);
		assert!(
			through[1] > bare[1] / 2,
			"and the pane is still there rather than gone: {through:?} against {bare:?}"
		);
	}

	#[test]
	fn the_nearer_of_two_panes_is_the_one_on_top() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the same two panes twice, with nothing swapped but how high each
		// hangs. The red one is registered first both times, so a build that
		// did not sort at all would draw them in that order in both frames and
		// answer the same way twice.
		let mut shot = |red: f32, green: f32| {
			let mut world = overhead_world();
			pane(&mut world, "test/red", rgb(0.9, 0.05, 0.05), red, Blend::Alpha);
			pane(&mut world, "test/green", rgb(0.05, 0.9, 0.05), green, Blend::Alpha);

			capture
				.shoot(&mut world)
				.expect("the capture renders")
				.pixel(SIZE.0 / 2, SIZE.1 / 2)
		};

		let red_above = shot(5.0, 3.0);
		let green_above = shot(3.0, 5.0);

		assert_eq!(
			dominant(red_above),
			0,
			"with the red pane nearer the camera, red is what is on top: {red_above:?}"
		);
		assert_eq!(
			dominant(green_above),
			1,
			"and with the green one nearer, green is: {green_above:?}"
		);
	}

	#[test]
	fn a_pane_is_sorted_by_where_its_geometry_is_and_not_by_its_origin() {
		let Some(mut capture) = capture() else {
			return;
		};

		// every built-in primitive is centered on its own origin, so a test
		// built out of them cannot tell the two apart at all. This one is a
		// quad lifted two units inside its own mesh, and the entity carrying it
		// hangs *lower* than the plain pane beside it - so the two answers are
		// opposite rather than merely different.
		let mut world = overhead_world();
		let mut lifted = mesh::quad();
		for vertex in &mut lifted.vertices {
			vertex.position[1] += 2.0;
		}
		let high = world.meshes.insert("test/lifted", lifted);

		let below = pane(&mut world, "test/red", rgb(0.9, 0.05, 0.05), 5.0, Blend::Alpha);
		assert!(below.is_some(), "the plain pane's material is registered");

		let green = world.materials.insert("test/green", Material {
			blend: Blend::Alpha,
			opacity: HALF,
			..Material::colored(rgb(0.05, 0.9, 0.05))
		});
		let entity = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 4.0, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(3.0, 1.0, 3.0),
		});
		world
			.entities
			.set_renderable(entity, Renderable::of(high, green, Vec3::ONE));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		// the green pane's entity is at four and the red one's at five, so by
		// origin the red one is nearer the overhead camera; but the green one's
		// geometry stands at six, so by geometry it is.
		assert_eq!(
			dominant(middle),
			1,
			"the pane whose triangles are nearer is the one on top, whatever its origin says: \
			 {middle:?}"
		);
	}

	#[test]
	fn two_panes_at_one_distance_do_not_hold_each_other_out() {
		let Some(mut capture) = capture() else {
			return;
		};

		// the same height, so the sort has nothing to say about which comes
		// first and the order falls to the registry: red was asked for first.
		// This is the one arrangement where the depth rule is the only thing
		// left, and it is what the rule exists for - a surface that wrote depth
		// would take the second pane away entirely rather than blend under it.
		let mut world = overhead_world();
		pane(&mut world, "test/red", rgb(0.9, 0.05, 0.05), 4.0, Blend::Alpha);
		pane(&mut world, "test/green", rgb(0.05, 0.9, 0.05), 4.0, Blend::Alpha);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(
			dominant(middle),
			1,
			"the second pane is composited over the first rather than failing a depth test \
			 against it: {middle:?}"
		);
		assert!(
			middle[0] > 40,
			"and the first is still under it rather than replaced: {middle:?}"
		);
	}

	#[test]
	fn a_pane_of_glass_casts_no_shadow() {
		let (Some((solid, world)), Some((glass, _))) =
			(cast_by(Blend::Opaque), cast_by(Blend::Alpha))
		else {
			return;
		};

		let cell = CASTER.1 / (2.0 * TILES);
		let middle = |column: f32| (column + 0.5).mul_add(cell, -CASTER.1 / 2.0);
		let under = Vec3::new(middle(0.0), CASTER.0, middle(0.0));

		let shaded = on_screen(&world, beneath(&world, under), SIZE);
		let beside = on_screen(&world, beneath(&world, under) + Vec3::new(0.0, 0.0, 4.0), SIZE);

		let lit = brightness(solid.pixel(beside.0, beside.1));
		let dark = brightness(solid.pixel(shaded.0, shaded.1));
		let through = brightness(glass.pixel(shaded.0, shaded.1));

		// the control: the same caster drawn solid does shade that spot, so
		// what is measured below is the mode and not the geometry.
		assert!(dark * 2 < lit, "drawn solid the caster shades the floor: {dark} against {lit}");
		assert!(
			through * 10 > lit * 8,
			"and drawn blended it casts nothing at all: {through} against {lit} beside it"
		);
	}

	#[test]
	fn a_blended_surface_reads_its_picture_and_its_opacity_both() {
		let Some(mut capture) = capture() else {
			return;
		};

		// a green pane over a red cube, so that which of the two a pixel is
		// mostly made of is a channel comparison rather than a threshold
		// somebody guessed.
		let mut shot = |opacity: f32| {
			let mut world = overhead_world();

			let cube = world.entities.spawn_at(Transform {
				position: Vec3::ZERO,
				rotation: Quat::IDENTITY,
				scale: Vec3::new(4.0, 2.0, 4.0),
			});
			world
				.entities
				.set_renderable(cube, Renderable::new(MeshId::CUBE, rgb(0.9, 0.05, 0.05)));

			let texture = world
				.textures
				.insert("test/holed green", holed_in([0x22, 0xFF, 0x22]));
			let material = world.materials.insert("test/glass", Material {
				blend: Blend::Alpha,
				opacity,
				..Material::textured(texture)
			});
			let pane = world.entities.spawn_at(Transform {
				position: Vec3::new(0.0, 2.0, 0.0),
				rotation: Quat::IDENTITY,
				scale: Vec3::new(3.0, 1.0, 3.0),
			});
			world
				.entities
				.set_renderable(pane, Renderable::of(MeshId::QUAD, material, Vec3::ONE));

			let image = capture
				.shoot(&mut world)
				.expect("the capture renders");

			// two quadrants of the picture, one solid and one nearly a hole.
			// Same material, same opacity, same cube under both: the alpha
			// channel is the only thing between them.
			let quarter = 3.0 / 4.0;
			let solid = on_screen(&world, Vec3::new(-quarter, 2.0, -quarter), SIZE);
			let thin = on_screen(&world, Vec3::new(quarter, 2.0, -quarter), SIZE);

			(image.pixel(solid.0, solid.1), image.pixel(thin.0, thin.1))
		};

		let (over_solid, over_thin) = shot(0.8);

		assert_eq!(
			dominant(over_solid),
			1,
			"where the picture is solid the pane is what is seen: {over_solid:?}"
		);
		assert_eq!(
			dominant(over_thin),
			0,
			"and where it is nearly a hole the cube under it is: {over_thin:?}"
		);

		// the same texel of the same picture at a different opacity. Without
		// the material's own number in the alpha, these two would be one pixel.
		let (fainter, _) = shot(0.35);

		assert!(
			distance(over_solid, fainter) > 20,
			"and the material's opacity moves the same texel: {over_solid:?} against 			 \
			 {fainter:?}"
		);
		assert!(
			u32::from(fainter[0]) > u32::from(over_solid[0]) + 20,
			"the fainter pane letting more of the cube through: {fainter:?} against 			 \
			 {over_solid:?}"
		);
	}

	#[test]
	fn a_debug_segment_behind_glass_is_seen_through_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// a pane between the camera and the segment, which is at the origin.
		// The segment writes depth and the pane does not, so what decides the
		// picture is only which of the two is recorded into the pass first.
		let mut glassed = with_a_line(0.0, false);
		let material = glassed.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: HALF,
			..Material::colored(rgb(0.9, 0.05, 0.05))
		});
		let sheet = glassed.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 0.0, 2.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(8.0, 8.0, 0.05),
		});
		glassed
			.entities
			.set_renderable(sheet, Renderable::of(MeshId::CUBE, material, Vec3::ONE));

		// the same pane over a world with no segment in it, so that what is
		// compared is the segment's own contribution rather than a column a
		// red pane is tinting. @ref `green_gain`.
		let mut paned = with_no_line();
		paned.materials.insert("test/glass", Material {
			blend: Blend::Alpha,
			opacity: HALF,
			..Material::colored(rgb(0.9, 0.05, 0.05))
		});
		let bare_sheet = paned.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 0.0, 2.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::new(8.0, 8.0, 0.05),
		});
		let glass = paned.materials.find("test/glass");
		paned
			.entities
			.set_renderable(bare_sheet, Renderable::of(MeshId::CUBE, glass, Vec3::ONE));

		let mut alone = with_a_line(0.0, false);
		let mut nothing = with_no_line();
		let through = green_gain(&mut capture, &mut glassed, &mut paned);
		let clear = green_gain(&mut capture, &mut alone, &mut nothing);

		let behind = capture
			.shoot(&mut glassed)
			.expect("the capture renders");
		let picture = capture
			.shoot(&mut alone)
			.expect("the capture renders again");
		let row = (SIZE.1 / 2 - 6..=SIZE.1 / 2 + 6)
			.max_by_key(|y| picture.pixel(SIZE.0 / 2, *y)[1])
			.expect("the range is not empty");

		assert!(clear > MARGIN, "the segment is there with nothing in front of it: {clear}");
		assert!(
			through * 10 < clear * 8,
			"and behind the pane it is dimmed by it rather than drawn over it: {through} \
			 against {clear}"
		);
		// the red half is still a pixel question rather than a column one: the
		// pane covers the whole row, so its own color is on every one of them
		// and there is nothing for anti-aliasing to spread.
		let (tinted, bare) = (behind.pixel(SIZE.0 / 2, row), picture.pixel(SIZE.0 / 2, row));

		assert!(
			u32::from(tinted[0]) > u32::from(bare[0]) + 40,
			"which is the pane's own color arriving on top of it: {tinted:?} against {bare:?}"
		);
	}

	#[test]
	fn a_prop_far_down_the_view_is_shadowed_by_a_further_cascade() {
		let Some(mut capture) = capture() else {
			return;
		};

		// down the floor rather than across it, so that what is on screen
		// spans more than one cascade instead of sitting inside the first.
		let mut world = shadowed_world();
		world.camera.position = Vec3::new(0.0, 2.5, 6.0);
		world.camera.target = Vec3::new(0.0, 0.0, -30.0);

		let near = Vec3::new(-2.0, 2.0, -4.0);
		let far = Vec3::new(-6.0, 5.0, -34.0);

		for (at, size) in [(near, 1.6), (far, 4.0)] {
			let prop = world.entities.spawn_at(Transform {
				position: at,
				rotation: Quat::IDENTITY,
				scale: Vec3::splat(size),
			});
			world
				.entities
				.set_renderable(prop, Renderable::new(MeshId::CUBE, Vec3::ONE));
		}

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		// the two casters are in different cascades: the near one is inside the
		// first cut and the far one is past the third.
		let cascades = shadow::fit(
			&world.render_camera(),
			world.aspect,
			world.light,
			shadow::DEFAULT_DISTANCE,
		);
		let depth = |point: Vec3| {
			let camera = world.render_camera();

			(point - camera.position).dot((camera.target - camera.position).normalize())
		};

		// the same rule the fragment applies: the first cut a depth fits under.
		let slice_of = |point: Vec3| {
			cascades
				.splits
				.iter()
				.position(|split| depth(point) <= *split)
				.unwrap_or(shadow::CASCADES)
		};

		assert!(
			slice_of(near) < slice_of(far),
			"the two casters are meant to be in different cascades, and they are both in {} at \
			 depths {} and {}",
			slice_of(near),
			depth(near),
			depth(far)
		);
		assert!(
			slice_of(far) < shadow::CASCADES,
			"and the further one is still inside the shadow distance, at {}",
			depth(far)
		);

		for (at, name) in [(near, "near"), (far, "far")] {
			let shaded = on_screen(&world, beneath(&world, at), SIZE);
			let beside = on_screen(&world, beneath(&world, at) + Vec3::new(0.0, 0.0, 4.0), SIZE);

			let dark = brightness(image.pixel(shaded.0, shaded.1));
			let lit = brightness(image.pixel(beside.0, beside.1));

			assert!(
				dark * 2 < lit,
				"the {name} prop's shadow at {shaded:?} reads {dark} against {lit} beside it"
			);
		}
	}

	#[test]
	fn a_stretched_entity_is_lit_by_the_normals_it_really_has() {
		let Some(mut capture) = capture() else {
			return;
		};

		// a sphere flattened almost to a disc and lit head on. The normals it
		// really has point at the camera almost everywhere, so the disc is
		// evenly lit out to its rim; the model matrix would carry them the
		// other way, leaving everything but the very center dark.
		let mut world = looking_world();
		let ball = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(2.0, 2.0, 0.05),
		});
		world
			.entities
			.set_renderable(ball, Renderable::new(MeshId::SPHERE, rgb(0.9, 0.9, 0.9)));

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		let (center_x, center_y) = (SIZE.0 / 2, SIZE.1 / 2);
		// the disc is a unit across in world units and the frame is about 2.73
		// half-heights, which puts its rim around forty-four pixels out.
		let center = brightness(image.pixel(center_x, center_y));
		let towards_the_rim = brightness(image.pixel(center_x + 30, center_y));

		assert!(center > 200, "the middle of the disc is lit at all: {center}");
		assert!(
			towards_the_rim * 10 > center * 7,
			"and so is two thirds of the way out, because a flattened sphere's normals still \
			 face the camera: {towards_the_rim} against {center}"
		);
	}
	/// A cube of one color a face, in the order the file writes them, with a
	/// chain whose every level is that same color.
	///
	/// Six colors rather than one, because what almost everything about a cube
	/// gets wrong is *which face*, and a cube of one color cannot tell.
	fn six_colors(side: u32) -> TextureData {
		const COLORS: [[f32; 3]; 6] = [
			[1.0, 0.0, 0.0],
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
			[1.0, 1.0, 0.0],
			[0.0, 1.0, 1.0],
			[1.0, 0.0, 1.0],
		];

		let count = TextureData::full_chain(side, side);
		let levels = (0..count)
			.map(|level| {
				let across = usize::try_from((side >> level.min(31)).max(1)).unwrap_or(1);

				COLORS
					.into_iter()
					.flat_map(|color| one_face(color, across * across))
					.collect()
			})
			.collect();

		TextureData {
			width: side,
			height: side,
			faces: 6,
			texel: Texel::Rgba16Float,
			levels,
		}
	}

	/// A cube whose every level is a different grey, brightest last.
	///
	/// What the six-colored one cannot show: which *level* a roughness reads.
	/// There the answer is the same color at every level, so a wrong level and
	/// a right one look alike.
	fn brightening(side: u32) -> TextureData {
		let count = TextureData::full_chain(side, side);
		let levels = (0..count)
			.map(|level| {
				let across = usize::try_from((side >> level.min(31)).max(1)).unwrap_or(1);
				let step = f32::from(u16::try_from(level).unwrap_or(0) + 1)
					/ f32::from(u16::try_from(count).unwrap_or(1));
				let texel: Vec<u8> = [step, step, step, 1.0]
					.into_iter()
					.flat_map(|value| eighth(value).to_le_bytes())
					.collect();

				texel.repeat(across * across * 6)
			})
			.collect();

		TextureData {
			width: side,
			height: side,
			faces: 6,
			texel: Texel::Rgba16Float,
			levels,
		}
	}

	/// A value in `0 ..= 1` as the sixteen bits the environment format holds.
	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "held inside nought and one by the callers, and assembled bit by bit rather 		          than converted"
	)]
	fn eighth(value: f32) -> u16 {
		if value <= 0.0 {
			return 0;
		}

		let exponent = value.log2().floor().clamp(-14.0, 15.0);
		let mantissa = ((value / exponent.exp2() - 1.0) * 1024.0)
			.round()
			.clamp(0.0, 1023.0);

		(((exponent as i32 + 15) as u16) << 10) | mantissa as u16
	}

	/// One face's worth of one color, opaque.
	fn one_face(color: [f32; 3], texels: usize) -> Vec<u8> {
		let texel: Vec<u8> = [color[0], color[1], color[2], 1.0]
			.into_iter()
			.flat_map(|value| half_bits(value).to_le_bytes())
			.collect();

		texel.repeat(texels)
	}

	/// Nought or one as the sixteen bits the environment format holds.
	fn half_bits(value: f32) -> u16 { if value > 0.5 { 0x3C00 } else { 0 } }

	/// A world looking straight down at a mirror-flat floor, nothing else lit.
	///
	/// The sun travels upwards and the ambient is black, so every pixel of the
	/// floor is the ambient specular term and nothing else - which is the term
	/// whose radiance an environment replaces. Looking straight down means what
	/// the floor reflects is straight up, which is one named face of the cube
	/// rather than a blend of several.
	fn mirror_world() -> World {
		let mut world = looking_world();
		world.light = Vec3::Y;
		world.ambient = Vec3::ZERO;
		world.post.tonemap = ToneMap::None;
		world.post.auto_exposure = false;
		world.post.exposure = 1.0;
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);
		world.camera.target = Vec3::ZERO;
		world
			.cvars
			.var(shadow::ENABLED, Value::Bool(false), "off for these pictures");

		let mirror = world.materials.insert("test/mirror", Material {
			metallic: 1.0,
			roughness: 0.0,
			..Material::DEFAULT
		});
		let across = view_across(&world);
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});
		world
			.entities
			.set_renderable(floor, Renderable::of(MeshId::QUAD, mirror, Vec3::ONE));

		world
	}

	#[test]
	fn a_mirror_under_no_light_at_all_reflects_the_environment_rather_than_black() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = mirror_world();
		let before = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let dark = brightness(before.pixel(SIZE.0 / 2, SIZE.1 / 2));

		let sky = world.textures.insert("test/sky", six_colors(4));
		world.sky = Sky::environment(sky);

		let after = capture
			.shoot(&mut world)
			.expect("the capture renders");
		let lit = after.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert!(
			dark < 8,
			"with no environment and no light a metal is black but for its highlight, which is \
			 the thing this card exists to stop: {dark}"
		);
		// the camera looks straight down at a floor facing up, so what the
		// middle of it reflects is +y, which is the third face and blue
		assert!(
			lit[2] > 180 && lit[0] < 40 && lit[1] < 40,
			"and with one it reflects the face the direction lands on, which is blue: {lit:?}"
		);
	}

	#[test]
	fn a_rough_metal_reads_further_down_the_chain_than_a_smooth_one() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = mirror_world();
		let sky = world.textures.insert("test/sky", brightening(8));
		world.sky = Sky::environment(sky);

		let smooth = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		// the same world with the floor roughened all the way. Each level of
		// this cube is a brighter grey than the one above it, so what the
		// picture says is *which level was read* - the one thing a cube of six
		// flat colors cannot tell, because there a wrong level and a right one
		// look alike.
		world.materials.insert("test/mirror", Material {
			metallic: 1.0,
			roughness: 1.0,
			..Material::DEFAULT
		});

		let rough = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert!(
			u32::from(rough[1]) > u32::from(smooth[1]) + 40,
			"a roughness of one reads the far end of the chain and a smooth surface reads the 			 near one: {rough:?} against {smooth:?}"
		);
	}

	#[test]
	fn a_world_whose_switch_is_off_draws_what_it_drew_before_there_were_any() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = mirror_world();
		// a cubemap sky naming nothing, which is the world this one is claimed
		// to draw: the switch turns off the *reflections*, and what is drawn
		// behind the world is the sky's own word either way
		world.sky = Sky::environment(TextureId::NONE);

		let before = capture
			.shoot(&mut world)
			.expect("the capture renders");

		let sky = world.textures.insert("test/sky", six_colors(4));
		world.sky = Sky::environment(sky);
		world
			.cvars
			.var(env::ENABLED, Value::Bool(false), "off for this picture");

		let after = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert_eq!(
			before.pixels, after.pixels,
			"the switch is the whole of what it says: a world naming an environment with it off \
			 draws the world that names none"
		);
	}

	#[test]
	fn a_cubemap_sky_is_drawn_behind_the_world_as_well_as_reflected_in_it() {
		let Some(mut capture) = capture() else {
			return;
		};

		// nothing standing in it, so every pixel of it is the sky
		let mut world = looking_world();
		world.clear = Vec3::ZERO;
		let sky = world.textures.insert("test/sky", six_colors(4));
		world.sky = Sky::environment(sky);

		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");
		// the camera looks along -z, which is the sixth face and magenta
		let middle = image.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert!(
			middle[0] > 180 && middle[1] < 40 && middle[2] > 180,
			"the middle of the picture looks along -z, and that face is magenta: {middle:?}"
		);

		world.sky = Sky::NONE;

		let none = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_ne!(middle, none, "and turning the sky off is a different picture");
	}

	#[test]
	fn a_cubemap_sky_naming_nothing_falls_back_to_the_three_colors() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = looking_world();
		world.clear = Vec3::ZERO;
		world.sky = Sky {
			kind: SkyKind::Cubemap,
			..Sky::gradient(Vec3::Z, Vec3::Y, Vec3::X)
		};

		let asked = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		world.sky = Sky::gradient(Vec3::Z, Vec3::Y, Vec3::X);

		let gradient = capture
			.shoot(&mut world)
			.expect("the capture renders")
			.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert_eq!(
			asked, gradient,
			"a word asking for a cubemap with none to show draws the gradient, because a black \
			 dome is nobody's idea of a sky"
		);
	}

	// ---- the rest of a material: its pictures, its glow, its turn, unlit ----

	/// Sets one of the frame's numbers, declared first the way its pass would.
	fn asked(world: &mut World, name: &str, value: &str) {
		world.cvars.var(name, Value::Float(1.0), "");
		world.cvars.set(name, value);
	}

	/// A picture of one value all over, stored as numbers.
	fn numbers_of(texel: [u8; 4]) -> TextureData {
		TextureData {
			width: 1,
			height: 1,
			faces: 1,
			texel: Texel::Rgba8Unorm,
			levels: vec![texel.to_vec()],
		}
	}

	/// Eight texels on a side, in a layout, each what `texel` says of its
	/// column and row.
	fn eight_by_eight<F>(texel: Texel, of: F) -> TextureData
	where
		F: Fn(u8, u8) -> [u8; 4],
	{
		TextureData {
			width: 8,
			height: 8,
			faces: 1,
			texel,
			levels: vec![
				(0..8_u8)
					.flat_map(|row| (0..8_u8).map(move |column| (column, row)))
					.flat_map(|(column, row)| of(column, row))
					.collect(),
			],
		}
	}

	/// Red squares on dark ones, as numbers: an occlusion picture of creases.
	fn creased() -> TextureData {
		eight_by_eight(Texel::Rgba8Unorm, |column, row| {
			if (column + row).is_multiple_of(2) {
				[255, 0, 0, 255]
			} else {
				[40, 0, 0, 255]
			}
		})
	}

	/// A picture whose red grows across it and whose green grows down it, so
	/// that the color that comes back says where a coordinate landed.
	fn ramps() -> TextureData {
		eight_by_eight(Texel::Rgba8Srgb, |column, row| [column * 32 + 16, row * 32 + 16, 64, 255])
	}

	/// Four colors in four quarters: red at the top left, green at the top
	/// right, blue at the bottom left and white at the bottom right, the top
	/// being the first row, which is where coordinates start.
	fn quarters() -> TextureData {
		eight_by_eight(Texel::Rgba8Srgb, |column, row| match (column < 4, row < 4) {
			| (true, true) => [255, 0, 0, 255],
			| (false, true) => [0, 255, 0, 255],
			| (true, false) => [0, 0, 255, 255],
			| (false, false) => [255, 255, 255, 255],
		})
	}

	/// A world looked at straight down, lit by the light from everywhere alone:
	/// the sun travels straight up, which lights nothing facing the camera.
	fn from_above() -> World {
		let mut world = looking_world();
		world.clear = Vec3::ZERO;
		world.ambient = Vec3::splat(0.8);
		world.light = Vec3::Y;
		world.camera.position = Vec3::new(0.0, HEIGHT, 0.01);

		world
	}

	/// A floor of a mesh and a material filling a square view from above.
	fn spread(world: &mut World, mesh: MeshId, material: MaterialId) -> EntityId {
		let across = view_across(world);
		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(across, 1.0, across),
		});

		world
			.entities
			.set_renderable(floor, Renderable::of(mesh, material, Vec3::ONE));

		floor
	}

	/// A world with a ball over a floor, a sun at a slant, the light from
	/// everywhere and the reflections and the share of the sky both on: every
	/// path a surface's numbers are read on, in one picture.
	fn showcase() -> World {
		let mut world = World::new();
		plainly(&mut world);
		world.clear = rgb(0.05, 0.07, 0.11);
		world.ambient = Vec3::splat(0.35);
		world.light = Vec3::new(-0.35, -1.0, -0.55).normalize();
		world.camera.position = Vec3::new(0.0, 2.5, 4.0);
		world.camera.target = Vec3::new(0.0, 0.6, 0.0);

		let floor = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::new(12.0, 1.0, 12.0),
		});

		world
			.entities
			.set_renderable(floor, Renderable::new(MeshId::QUAD, rgb(0.6, 0.6, 0.58)));

		world
	}

	/// Stands a ball of a material in the middle of a showcase.
	fn ball(world: &mut World, material: MaterialId) {
		let ball = world.entities.spawn_at(Transform {
			position: Vec3::new(0.0, 0.8, 0.0),
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(1.4),
		});

		world
			.entities
			.set_renderable(ball, Renderable::of(MeshId::SPHERE, material, Vec3::ONE));
	}

	/// How far apart two pictures are: the largest difference in any channel
	/// of any pixel, and how many pixels differ by more than one.
	fn apart(one: &Image, other: &Image) -> (u8, usize) {
		one.pixels
			.chunks(4)
			.zip(other.pixels.chunks(4))
			.map(|(left, right)| {
				left.iter()
					.zip(right)
					.map(|(a, b)| a.abs_diff(*b))
					.max()
					.unwrap_or(0)
			})
			.fold((0, 0), |(worst, many), off| (worst.max(off), many + usize::from(off > 1)))
	}

	/// A byte of the picture as the light it stands for.
	fn linear(byte: u8) -> f32 {
		let fraction = f32::from(byte) / 255.0;

		if fraction <= 0.04045 {
			fraction / 12.92
		} else {
			((fraction + 0.055) / 1.055).powf(2.4)
		}
	}

	/// The byte a light comes out as, by the curve every screen reads.
	fn encoded(light: f32) -> u8 {
		let curved = if light <= 0.003_130_8 {
			light * 12.92
		} else {
			1.055_f32.mul_add(light.powf(1.0 / 2.4), -0.055)
		};

		pixel((curved * 255.0).round(), 256.0)
			.try_into()
			.unwrap_or(u8::MAX)
	}

	#[test]
	fn a_finish_picture_of_one_value_is_the_two_numbers_it_holds() {
		// blue multiplies how metal the surface is and green how rough, so a
		// picture holding one value all over draws what a material holding
		// those two numbers draws - lit by the sun, by the light from
		// everywhere, in the reflections, which read the roughness the pass
		// before the scene wrote, and in the share of the sky. The two bytes are
		// ones a division by 255 and a multiply by its inverse read back as the
		// same float, which is every way a device turns a byte into a number.
		let (green, blue) = (89_u8, 130_u8);
		let (Some(mut first), Some(mut second), Some(mut third)) =
			(capture(), capture(), capture())
		else {
			return;
		};
		let color = rgb(0.9, 0.6, 0.3);

		let mut pictured = showcase();
		let finish = pictured
			.textures
			.insert("test/finish", numbers_of([255, green, blue, 255]));
		let material = pictured
			.materials
			.insert("test/finished", Material {
				base_color: color,
				metallic: 1.0,
				roughness: 1.0,
				finish,
				..Material::DEFAULT
			});
		ball(&mut pictured, material);

		let mut numbered = showcase();
		let material = numbered
			.materials
			.insert("test/finished", Material {
				base_color: color,
				metallic: f32::from(blue) / 255.0,
				roughness: f32::from(green) / 255.0,
				..Material::DEFAULT
			});
		ball(&mut numbered, material);

		let mut bare = showcase();
		let material = bare.materials.insert("test/finished", Material {
			base_color: color,
			metallic: 1.0,
			roughness: 1.0,
			..Material::DEFAULT
		});
		ball(&mut bare, material);

		let picture = first.shoot(&mut pictured).expect("it renders");
		let numbers = second.shoot(&mut numbered).expect("it renders");
		let neither = third.shoot(&mut bare).expect("it renders");

		assert!(picture.pixels == numbers.pixels, "the picture is its two numbers, to the byte");
		assert!(
			apart(&picture, &neither).1 > 1000,
			"and a picture that read nothing would be a picture of a rough metal: {:?}",
			apart(&picture, &neither)
		);
	}

	#[test]
	fn an_occlusion_picture_at_a_strength_of_nought_is_no_picture() {
		let (Some(mut first), Some(mut second), Some(mut third)) =
			(capture(), capture(), capture())
		else {
			return;
		};
		let shoot = |capture: &mut Capture, strength: Option<f32>| {
			let mut world = showcase();
			let occlusion = world.textures.insert("test/creased", creased());
			let material = world.materials.insert("test/creased", Material {
				base_color: rgb(0.8, 0.8, 0.8),
				occlusion: if strength.is_some() { occlusion } else { TextureId::NONE },
				occlusion_strength: strength.unwrap_or(1.0),
				..Material::DEFAULT
			});
			ball(&mut world, material);

			capture.shoot(&mut world).expect("it renders")
		};

		let nought = shoot(&mut first, Some(0.0));
		let none = shoot(&mut second, None);
		let whole = shoot(&mut third, Some(1.0));

		assert!(
			nought.pixels == none.pixels,
			"one plus nought times anything is one, to the byte"
		);
		assert!(apart(&whole, &none).1 > 100, "and at a strength of one the creases show");
	}

	#[test]
	fn an_occlusion_picture_darkens_the_light_from_everywhere_by_its_own_value() {
		// the light from everywhere as the only light, and the share of the sky
		// the frame works out off, so that nothing but the picture occludes: a
		// picture of a half, at a strength of one, leaves half of it.
		let (Some(mut first), Some(mut second)) =
			(capture_of(SQUARE, SQUARE), capture_of(SQUARE, SQUARE))
		else {
			return;
		};
		let value = 128_u8;
		let shoot = |capture: &mut Capture, pictured: bool| {
			let mut world = from_above();
			asked(&mut world, crate::occlusion::STRENGTH, "0");

			let occlusion = world
				.textures
				.insert("test/half", numbers_of([value, 0, 0, 255]));
			let material = world.materials.insert("test/half", Material {
				base_color: rgb(0.9, 0.9, 0.9),
				occlusion: if pictured { occlusion } else { TextureId::NONE },
				..Material::DEFAULT
			});
			spread(&mut world, MeshId::QUAD, material);

			capture.shoot(&mut world).expect("it renders")
		};

		let dimmed = shoot(&mut first, true);
		let open = shoot(&mut second, false);
		let share = f32::from(value) / 255.0;

		for (x, y) in
			[(SQUARE / 2, SQUARE / 2), (SQUARE / 4, SQUARE / 3), (SQUARE * 3 / 4, SQUARE / 5)]
		{
			let (under, over) = (dimmed.pixel(x, y), open.pixel(x, y));

			for channel in 0..3 {
				let ratio = linear(under[channel]) / linear(over[channel]).max(1e-6);

				assert!(
					(ratio - share).abs() < 0.015,
					"at {x},{y} the light from everywhere is {share} of what it was, and is \
					 {ratio}: {under:?} against {over:?}"
				);
			}
		}
	}

	#[test]
	fn an_occlusion_picture_leaves_what_a_sun_and_a_lamp_send_alone() {
		// the exchange format's rule: what it darkens is the light arriving
		// from everywhere, and nothing a light sends. With none of that light
		// in the world, the picture changes nothing, to the byte.
		let (Some(mut first), Some(mut second)) = (capture(), capture()) else {
			return;
		};
		let shoot = |capture: &mut Capture, pictured: bool| {
			let mut world = showcase();
			world.ambient = Vec3::ZERO;

			let lamp = world
				.entities
				.spawn_at(Transform::at(Vec3::new(1.2, 1.6, 1.2)));
			world
				.entities
				.set_light(lamp, colby_core::abi::Light::point(rgb(1.0, 0.8, 0.6), 3.0, 6.0));

			let occlusion = world.textures.insert("test/creased", creased());
			let material = world.materials.insert("test/creased", Material {
				occlusion: if pictured { occlusion } else { TextureId::NONE },
				..Material::DEFAULT
			});
			ball(&mut world, material);

			capture.shoot(&mut world).expect("it renders")
		};

		let pictured = shoot(&mut first, true);
		let bare = shoot(&mut second, false);

		assert!(pictured.pixels == bare.pixels, "a sun and a lamp are not occluded by it");
	}

	#[test]
	fn a_surface_giving_off_light_in_the_dark_is_the_color_it_gives_off() {
		// nothing lights the floor, so all it sends is what it gives off: which
		// is what an unlit floor of that color sends whatever lights it, what
		// half of it at twice the strength sends, and the byte the color is.
		//
		// Three colors whose bytes are nowhere near a half: the unlit color
		// reaches the fragment stage interpolated and the glow out of a uniform,
		// and a device may part the two by an ulp, which a color at 224.6 - three
		// quarters - carries across the rounding on one of them.
		let given = rgb(0.25, 0.328_125, 0.921_875);
		let mut captures = [(); 4].map(|()| capture_of(SQUARE, SQUARE));
		let shoot = |capture: &mut Option<Capture>, material: Material| {
			let mut world = from_above();
			world.ambient = Vec3::ZERO;
			let material = world.materials.insert("test/lit", material);
			spread(&mut world, MeshId::QUAD, material);

			capture
				.as_mut()
				.map(|capture| capture.shoot(&mut world).expect("it renders"))
		};

		let [Some(glowing), Some(doubled), Some(unlit), Some(dark)] = [
			shoot(&mut captures[0], Material {
				base_color: Vec3::ZERO,
				emissive: given,
				..Material::DEFAULT
			}),
			shoot(&mut captures[1], Material {
				base_color: Vec3::ZERO,
				emissive: given * 0.5,
				emissive_strength: 2.0,
				..Material::DEFAULT
			}),
			shoot(&mut captures[2], Material {
				base_color: given,
				unlit: true,
				..Material::DEFAULT
			}),
			shoot(&mut captures[3], Material {
				base_color: Vec3::ZERO,
				..Material::DEFAULT
			}),
		] else {
			return;
		};

		assert!(glowing.pixels == unlit.pixels, "the light it gives off is its color, unlit");
		assert!(glowing.pixels == doubled.pixels, "and the strength multiplies the color");
		assert_ne!(glowing.pixel(SQUARE / 2, SQUARE / 2), dark.pixel(SQUARE / 2, SQUARE / 2));

		let middle = glowing.pixel(SQUARE / 2, SQUARE / 2);

		for (channel, light) in given.to_array().into_iter().enumerate() {
			assert!(
				middle[channel].abs_diff(encoded(light)) <= 1,
				"channel {channel} is the byte {light} is, {} against {}",
				middle[channel],
				encoded(light)
			);
		}

		// and in a fog: the light it gives off is fogged as the unlit color is,
		// because both go into the one fog, after everything that lights them
		let fogged = |material: Material| {
			let mut capture = capture_of(SQUARE, SQUARE)?;
			let mut world = from_above();
			world.ambient = Vec3::ZERO;
			world.post.fog = rgb(0.3, 0.4, 0.5);
			world.post.fog_density = 0.15;
			let material = world.materials.insert("test/fogged", material);
			spread(&mut world, MeshId::QUAD, material);

			Some(capture.shoot(&mut world).expect("it renders"))
		};
		let (Some(glowing), Some(unlit)) = (
			fogged(Material {
				base_color: Vec3::ZERO,
				emissive: given,
				..Material::DEFAULT
			}),
			fogged(Material {
				base_color: given,
				unlit: true,
				..Material::DEFAULT
			}),
		) else {
			return;
		};

		// within a level rather than to the byte: the fog puts every pixel at a
		// value of its own, and at one of them the ulp the two may part by lands
		// on a rounding
		let (worst, _) = apart(&glowing, &unlit);

		assert!(worst <= 1, "and in a fog, fogged alike: {worst}");
		assert!(
			glowing.pixel(SQUARE / 2, SQUARE / 2)[0].abs_diff(encoded(given.x)) > 3,
			"by a fog thick enough to move the byte"
		);
	}

	#[test]
	fn the_share_of_the_sky_and_an_occlusion_picture_are_taken_by_the_smaller() {
		// both are pictures of the same crease: where the frame's share is the
		// smaller, a floor under an occlusion picture is the floor without one,
		// and where the picture is, it is the floor with the share off, to the
		// byte. A product would be darker than both under every crease.
		let shoot = |share: bool, pictured: bool| {
			let mut capture = capture()?;
			let mut world = showcase();
			asked(&mut world, crate::occlusion::STRENGTH, if share { "1" } else { "0" });
			// one sample a pixel: at four, a pixel on the ball's edge mixes the
			// ball, lit with the share in one picture and without it in the
			// other, with the floor, and is rightly neither
			asked(&mut world, MSAA, "1");

			let occlusion = world
				.textures
				.insert("test/three-quarters", numbers_of([191, 0, 0, 255]));
			let floor = world.materials.insert("test/floor", Material {
				base_color: rgb(0.6, 0.6, 0.58),
				occlusion: if pictured { occlusion } else { TextureId::NONE },
				..Material::DEFAULT
			});
			let first = world.entities.iter().map(|(id, ..)| id).next();

			if let Some(id) = first {
				world
					.entities
					.set_renderable(id, Renderable::of(MeshId::QUAD, floor, Vec3::ONE));
			}

			// sitting on the floor rather than over it, for a crease at its foot
			let ball = world.entities.spawn_at(Transform {
				position: Vec3::new(0.0, 0.7, 0.0),
				rotation: Quat::IDENTITY,
				scale: Vec3::splat(1.4),
			});

			world
				.entities
				.set_renderable(ball, Renderable::new(MeshId::SPHERE, Vec3::ONE));

			Some(capture.shoot(&mut world).expect("it renders"))
		};
		let (Some(both), Some(share), Some(picture)) =
			(shoot(true, true), shoot(true, false), shoot(false, true))
		else {
			return;
		};
		let mut taken = (0, 0);

		for ((one, left), right) in both
			.pixels
			.chunks(4)
			.zip(share.pixels.chunks(4))
			.zip(picture.pixels.chunks(4))
		{
			if one == left {
				taken.0 += 1;
			} else {
				assert_eq!(one, right, "a pixel is the share's or the picture's, not darker");
				taken.1 += 1;
			}
		}

		assert!(taken.1 > 1000, "the picture is the smaller over the open floor: {taken:?}");
		assert!(
			apart(&both, &picture).1 > 100,
			"and the share the smaller at the ball's foot, over enough of it to see: {taken:?}"
		);
	}

	#[test]
	fn an_unlit_surface_is_its_color_whatever_lights_it() {
		// a ball against nothing: lit by a sun, a lamp and the light from
		// everywhere, or by none of them, it is the same bytes, and those are
		// its color's.
		let color = rgb(0.2, 0.6, 0.35);
		let (Some(mut first), Some(mut second)) = (capture(), capture()) else {
			return;
		};
		let shoot = |capture: &mut Capture, lit: bool| {
			let mut world = looking_world();
			world.clear = Vec3::ZERO;
			world.ambient = if lit { Vec3::splat(0.7) } else { Vec3::ZERO };
			world.light = if lit { Vec3::NEG_Z } else { Vec3::Z };

			if lit {
				let lamp = world
					.entities
					.spawn_at(Transform::at(Vec3::new(0.6, 0.6, 1.5)));

				world
					.entities
					.set_light(lamp, colby_core::abi::Light::point(Vec3::ONE, 4.0, 5.0));
			}

			let material = world.materials.insert("test/unlit", Material {
				base_color: color,
				unlit: true,
				..Material::DEFAULT
			});
			let ball = world.entities.spawn_at(Transform::IDENTITY);

			world
				.entities
				.set_renderable(ball, Renderable::of(MeshId::SPHERE, material, Vec3::ONE));

			capture.shoot(&mut world).expect("it renders")
		};

		let lit = shoot(&mut first, true);
		let dark = shoot(&mut second, false);
		let middle = lit.pixel(SIZE.0 / 2, SIZE.1 / 2);

		assert!(lit.pixels == dark.pixels, "no light reaches it and none is needed");

		for (channel, light) in color.to_array().into_iter().enumerate() {
			assert!(
				middle[channel].abs_diff(encoded(light)) <= 1,
				"channel {channel} is the byte {light} is, {} against {}",
				middle[channel],
				encoded(light)
			);
		}
	}

	/// A floor of a picture, looked at from above, its coordinates moved by a
	/// material and by nothing else.
	fn moved_floor(picture: TextureData, material: Material, mesh: MeshData) -> Option<Image> {
		let mut capture = capture_of(SQUARE, SQUARE)?;
		let mut world = from_above();
		world.ambient = Vec3::ONE;

		let albedo = world.textures.insert("test/moved", picture);
		let mesh = world.meshes.insert("test/moved", mesh);
		let material = world
			.materials
			.insert("test/moved", Material { albedo, ..material });
		spread(&mut world, mesh, material);

		Some(capture.shoot(&mut world).expect("it renders"))
	}

	#[test]
	fn the_exchange_formats_own_example_shows_the_lower_left_quarter() {
		// the specification's example: an offset of nought and one, a turn of a
		// quarter and a scale of a half "utilizes only the lower left quadrant
		// of the source image, rotated clockwise". Turned the other way the same
		// numbers show the top right quarter, which is what the specification's
		// own matrix read as it is written would do; this is the oracle that
		// decides between the two.
		let turn = core::f32::consts::FRAC_PI_2;
		let example = |rotation: f32| {
			moved_floor(
				quarters(),
				Material {
					uv_offset: Vec2::new(0.0, 1.0),
					uv_rotation: rotation,
					uv_scale: Vec2::splat(0.5),
					..Material::DEFAULT
				},
				mesh::quad(),
			)
		};
		let (Some(asked), Some(other)) = (example(turn), example(-turn)) else {
			return;
		};

		// by which channel is largest rather than by the byte: the light from
		// everywhere sends a little of every color back off any surface
		for x in (SQUARE / 5..SQUARE * 4 / 5).step_by(16) {
			for y in (SQUARE / 5..SQUARE * 4 / 5).step_by(16) {
				assert_eq!(dominant(asked.pixel(x, y)), 2, "blue at {x},{y}, the bottom left");
				assert_eq!(dominant(other.pixel(x, y)), 1, "and green turned the other way");
			}
		}
	}

	/// Coordinates scaled, turned and moved on the way in, the three things the
	/// shader does to them on the way through: the exchange format's order and
	/// its sense of a turn.
	fn turned_by_hand(uv: Vec2, scale: Vec2, rotation: f32, offset: Vec2) -> Vec2 {
		let (sin, cos) = rotation.sin_cos();
		let scaled = uv * scale;

		offset
			+ Vec2::new(
				sin.mul_add(scaled.y, cos * scaled.x),
				cos.mul_add(scaled.y, -sin * scaled.x),
			)
	}

	/// The same grid with its coordinates moved on the way in.
	fn grid_moved<F>(side: u16, moved: F) -> MeshData
	where
		F: Fn(Vec2) -> Vec2,
	{
		let mut data = grid(side);

		// after the tangents, which stay the unmoved grid's, so that the two
		// floors differ in their coordinates and in nothing else
		for vertex in &mut data.vertices {
			vertex.uv = moved(Vec2::from_array(vertex.uv)).to_array();
		}

		data
	}

	#[test]
	fn a_turned_picture_is_the_picture_with_its_coordinates_turned_by_hand() {
		// scaled, turned, then moved, in the shader, against a mesh whose
		// coordinates were put through the same three on the way in: the two
		// sums round a hair apart, and that is all, where the same numbers
		// turned the other way are another picture.
		// a stretch of 1.3 against 0.7: at 1.5 against 0.75 the two floors part by
		// two levels at one pixel under one of the two APIs, on the steepest step
		// of the ramp, and this one parts by one at most under both
		let (scale, rotation, offset) = (Vec2::new(1.3, 0.7), 0.6_f32, Vec2::new(0.3, -0.2));
		let by_hand =
			|sign: f32| move |uv: Vec2| turned_by_hand(uv, scale, rotation * sign, offset);
		let (Some(shader), Some(hand), Some(other)) = (
			moved_floor(
				ramps(),
				Material {
					uv_scale: scale,
					uv_rotation: rotation,
					uv_offset: offset,
					..Material::DEFAULT
				},
				grid(4),
			),
			moved_floor(ramps(), Material::DEFAULT, grid_moved(4, by_hand(1.0))),
			moved_floor(ramps(), Material::DEFAULT, grid_moved(4, by_hand(-1.0))),
		) else {
			return;
		};

		let (worst, many) = apart(&shader, &hand);

		assert!(worst <= 1 && many == 0, "within a byte everywhere: {worst}, {many}");
		assert!(apart(&shader, &other).1 > 1000, "and turned the other way is another picture");
	}

	#[test]
	fn a_picture_on_the_second_set_laid_out_as_the_first_is_the_same_picture() {
		// a mesh whose second set is its first, read by an occlusion and a glow
		// on either set: one picture, to the byte. And the same mesh with its
		// second set squeezed into a corner is another picture for each of the
		// two read from it alone, or that one's flag is read by nothing.
		let shoot = |occlusion_uv2: bool, glow_uv2: bool, squeezed: bool| {
			let mut capture = capture_of(SQUARE, SQUARE)?;
			let mut world = from_above();
			let mut data = grid(4);
			// a tenth of the first set, or all of it, which is the first set to
			// the bit
			let share = if squeezed { 0.1 } else { 1.0 };

			data.paint = data
				.vertices
				.iter()
				.map(|vertex| PaintVertex::new(Vec4::ONE, Vec2::from_array(vertex.uv) * share))
				.collect();

			let mesh = world.meshes.insert("test/second", data);
			let occlusion = world.textures.insert("test/creased", creased());
			let glow = world.textures.insert("test/ramps", ramps());
			let material = world.materials.insert("test/second", Material {
				occlusion,
				occlusion_uv2,
				glow,
				glow_uv2,
				emissive: Vec3::splat(0.3),
				..Material::DEFAULT
			});
			spread(&mut world, mesh, material);

			Some(capture.shoot(&mut world).expect("it renders"))
		};

		let (Some(first), Some(second), Some(occluded), Some(glowing)) = (
			shoot(false, false, false),
			shoot(true, true, false),
			shoot(true, false, true),
			shoot(false, true, true),
		) else {
			return;
		};

		assert!(first.pixels == second.pixels, "the same coordinates are the same picture");
		assert!(apart(&first, &occluded).1 > 1000, "the occlusion on other coordinates");
		assert!(apart(&first, &glowing).1 > 1000, "and the glow on other coordinates");
	}

	#[test]
	fn a_turned_cutout_casts_the_shadow_of_the_cutout_turned_by_hand() {
		// the pass that draws the shadows reads the same turn the picture does,
		// or a fence's shadow would have holes where the fence has none
		let (scale, rotation, offset) = (Vec2::new(2.0, 1.25), 0.9_f32, Vec2::new(0.1, 0.35));
		let by_hand = move |uv: Vec2| turned_by_hand(uv, scale, rotation, offset);
		// the arrangement a cutout's own shadow test uses: overhead, with the
		// shadow its own width to one side of it
		let shoot = |material: Material, mesh: MeshData| {
			let mut capture = capture()?;
			let mut world = shadowed_world();
			world.camera.position = Vec3::new(0.0, 9.0, 0.01);
			world.camera.target = Vec3::ZERO;

			let holes = world.textures.insert("test/holed", holed());
			let mesh = world.meshes.insert("test/fence", mesh);
			let material = world.materials.insert("test/fence", Material {
				albedo: holes,
				blend: Blend::Mask,
				..material
			});
			let fence = world.entities.spawn_at(Transform {
				position: Vec3::new(0.0, CASTER.0, 0.0),
				rotation: Quat::IDENTITY,
				scale: Vec3::new(CASTER.1, 1.0, CASTER.1),
			});

			world
				.entities
				.set_renderable(fence, Renderable::of(mesh, material, Vec3::ONE));

			Some(capture.shoot(&mut world).expect("it renders"))
		};
		let turned = Material {
			uv_scale: scale,
			uv_rotation: rotation,
			uv_offset: offset,
			..Material::DEFAULT
		};

		let (Some(shader), Some(hand), Some(unmoved)) = (
			shoot(turned, grid(4)),
			shoot(Material::DEFAULT, grid_moved(4, by_hand)),
			shoot(Material::DEFAULT, grid(4)),
		) else {
			return;
		};

		let (_, many) = apart(&shader, &hand);

		assert!(many < 20, "the same holes in the fence and its shadow: {many} pixels apart");
		assert!(apart(&shader, &unmoved).1 > 500, "where the unturned cutout is another picture");
	}

	#[test]
	fn both_shaders_turn_a_picture_the_same_way_and_read_one_block() {
		// the scene's shader and the shadows' each hold the turn and the block
		// it is read from, because a module cannot include another one; a
		// cutout turned one way whose shadow is turned another is the bug this
		// stops.
		let text = |source: &'static str, from: &str, to: &str| {
			let start = source
				.find(from)
				.unwrap_or_else(|| panic!("`{from}` is in the shader"));
			let rest = source.get(start..).unwrap_or_default();
			let end = rest
				.find(to)
				.unwrap_or_else(|| panic!("and so is the `{to}` after it"));

			rest.get(..end)
				.unwrap_or_default()
				.lines()
				.map(str::trim)
				.filter(|line| !line.starts_with("//"))
				.collect::<Vec<_>>()
				.join("\n")
		};
		let scene = include_str!("shader.wgsl");
		let shadow = include_str!("shadow.wgsl");

		assert_eq!(
			text(scene, "fn turned(", "\n}"),
			text(shadow, "fn turned(", "\n}"),
			"one turn in both"
		);
		assert_eq!(
			text(scene, "struct Finish {", "};"),
			text(shadow, "struct Finish {", "};"),
			"and one block"
		);
		assert!(
			scene.contains("@group(1) @binding(10) var<uniform> finish: Finish;")
				&& shadow.contains("@group(2) @binding(10) var<uniform> finish: Finish;"),
			"at the binding the material's layout puts it"
		);
	}

	#[test]
	fn a_materials_numbers_changed_in_place_are_drawn_on_the_next_frame() {
		// the uniform is written when the material's revision moves, the way
		// its pictures are bound again when theirs do
		let Some(mut capture) = capture_of(SQUARE, SQUARE) else {
			return;
		};
		let mut world = from_above();
		let material = world
			.materials
			.insert("test/moving", Material::DEFAULT);
		spread(&mut world, MeshId::QUAD, material);

		let before = capture.shoot(&mut world).expect("it renders");

		if let Some(held) = world.materials.get_mut(material) {
			held.emissive = rgb(0.5, 0.0, 0.0);
		}

		let glowing = capture.shoot(&mut world).expect("it renders");

		if let Some(held) = world.materials.get_mut(material) {
			held.emissive = Vec3::ZERO;
		}

		let after = capture.shoot(&mut world).expect("it renders");

		assert_ne!(glowing.pixel(SQUARE / 2, SQUARE / 2), before.pixel(SQUARE / 2, SQUARE / 2));
		assert!(after.pixels == before.pixels, "and put back, it is the picture it was");
	}

	#[test]
	fn a_finish_picture_uploaded_again_is_read_again() {
		// the group holds a view of one texture, and a picture reloaded under
		// its name is a new one: every picture a material names is watched,
		// not only the first two
		let Some(mut capture) = capture() else {
			return;
		};
		let mut world = showcase();
		let finish = world
			.textures
			.insert("test/finish", numbers_of([255, 255, 255, 255]));
		let material = world.materials.insert("test/finished", Material {
			metallic: 1.0,
			roughness: 1.0,
			finish,
			..Material::DEFAULT
		});
		ball(&mut world, material);

		let rough = capture.shoot(&mut world).expect("it renders");

		world
			.textures
			.insert("test/finish", numbers_of([255, 40, 255, 255]));

		let smooth = capture.shoot(&mut world).expect("it renders");

		assert!(apart(&rough, &smooth).1 > 500, "the new picture is read");
	}
}
