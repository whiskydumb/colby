//! Rendering a world into memory instead of into a window.
//!
//! This exists so that "does it actually look right" stops being a question
//! only a person at the screen can answer. It renders with the same [`Scene`]
//! the window uses, reads the pixels back, and hands over an [`Image`] that a
//! test can make assertions about - or that can be written out as a PNG and
//! looked at.

use colby_core::{Result, abi::World, debug, err, glam::Vec3};
use wgpu::{
	Backends, BufferDescriptor, BufferUsages, DeviceDescriptor, ExperimentalFeatures, Extent3d,
	Features, Instance, InstanceDescriptor, Limits, MapMode, MemoryHints, PollType,
	PowerPreference, RequestAdapterOptions, TexelCopyBufferInfo, TexelCopyBufferLayout,
	TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
};

use crate::{image::Image, overlay::Overlay, scene::Scene};

/// The color format captures render into.
///
/// The same family the window uses, so that the numbers a test reads back are
/// the numbers that would reach the screen.
const CAPTURE_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;

/// Copying a texture to a buffer wants each row aligned to this.
const ROW_ALIGNMENT: u32 = 256;

/// An offscreen target and everything needed to draw into it.
pub struct Capture {
	scene: Scene,
	width: u32,
	height: u32,
	color: wgpu::Texture,
	readback: wgpu::Buffer,
	/// Bytes per row in the readback buffer, padded up to [`ROW_ALIGNMENT`].
	padded_stride: u32,
}

impl Capture {
	/// Builds an offscreen renderer, if this machine has a GPU to build it on.
	///
	/// @param width - the image width in pixels
	/// @param height - the image height in pixels
	/// @return the capture, or `None` when no adapter could be found
	pub fn new(width: u32, height: u32) -> Result<Option<Self>> {
		pollster::block_on(Self::create(width, height))
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
		world.aspect = self.aspect();

		let view = self
			.color
			.create_view(&TextureViewDescriptor::default());

		self.scene.render(&view, world);

		for overlay in overlays {
			overlay.draw(self.scene.device(), self.scene.queue(), &view, self.width, self.height);
		}

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
	pub const fn format() -> TextureFormat { CAPTURE_FORMAT }

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

		drop(view);
		self.readback.unmap();

		Ok(Image {
			width: self.width,
			height: self.height,
			pixels,
		})
	}

	/// The async half of [`new`](Self::new).
	async fn create(width: u32, height: u32) -> Result<Option<Self>> {
		let (width, height) = (width.max(1), height.max(1));
		let instance = Instance::new(InstanceDescriptor {
			backends: Backends::DX12 | Backends::VULKAN,
			..InstanceDescriptor::new_without_display_handle()
		});

		let Ok(adapter) = instance
			.request_adapter(&RequestAdapterOptions {
				power_preference: PowerPreference::HighPerformance,
				..Default::default()
			})
			.await
		else {
			return Ok(None);
		};

		let info = adapter.get_info();
		debug!(adapter = %info.name, backend = ?info.backend, "capture adapter selected");

		let (device, queue) = adapter
			.request_device(&DeviceDescriptor {
				label: Some("colby capture"),
				required_features: Features::empty(),
				required_limits: Limits::default(),
				experimental_features: ExperimentalFeatures::disabled(),
				memory_hints: MemoryHints::Performance,
				trace: wgpu::Trace::Off,
			})
			.await
			.map_err(|error| err!(Graphics("requesting a capture device: {error}")))?;

		let color = device.create_texture(&TextureDescriptor {
			label: Some("capture"),
			size: Extent3d { width, height, depth_or_array_layers: 1 },
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: CAPTURE_FORMAT,
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

		let scene = Scene::new(device, queue, CAPTURE_FORMAT, width, height)?;

		Ok(Some(Self {
			scene,
			width,
			height,
			color,
			readback,
			padded_stride,
		}))
	}
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
		abi::{Material, MeshId, Renderable, Texel, TextureData, Transform},
		glam::Quat,
	};

	use super::*;

	/// How big the test captures are. Small enough to be quick, large enough
	/// that a sample well inside a shape is unambiguous.
	const SIZE: (u32, u32) = (320, 240);

	/// The side of the one capture that has to be square. @ref
	/// [`a_texture_reaches_the_screen_the_way_up_its_coordinates_say`].
	const SQUARE: u32 = 256;

	/// How far above the origin the overhead camera sits.
	const HEIGHT: f32 = 5.0;

	/// A world with a camera looking at the origin from `+z`, and a light
	/// traveling the same way the camera does.
	///
	/// With no ambient at all, a surface facing the camera is fully lit and one
	/// facing away is black. That is what makes the winding test decisive.
	fn looking_world() -> World {
		let mut world = World::new();
		world.clear = rgb(0.0, 0.0, 0.2);
		world.ambient = Vec3::ZERO;
		world.light = Vec3::NEG_Z;
		world.camera.position = Vec3::new(0.0, 0.0, 5.0);
		world.camera.target = Vec3::ZERO;

		world
	}

	/// A capture, or `None` with a note when this machine has no GPU.
	fn capture() -> Option<Capture> {
		match Capture::new(SIZE.0, SIZE.1) {
			| Ok(capture) => capture,
			| Err(error) => panic!("building the capture failed: {error}"),
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
		let tiny = colby_asset::obj::import(
			"v -0.02 -0.02 0\nv 0.02 -0.02 0\nv 0.0 -0.015 0\nf 1 2 3\n",
		)
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
		let source = include_str!("shader.wgsl").replace(
			"return vec4<f32>(direct + indirect, 1.0);",
			"return vec4<f32>(0.0, 0.0, 1.0, 1.0);",
		);

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

	/// A two-by-two image: red, green over blue, white, with its whole chain.
	///
	/// Built by hand rather than decoded, because the decoder is
	/// `colby_asset`'s to test. What this one is for is the other end: whether
	/// those texels come out of the screen where the coordinates say they
	/// should.
	fn quadrants() -> TextureData {
		let base = vec![
			0xFF, 0x00, 0x00, 0xFF, // red
			0x00, 0xFF, 0x00, 0xFF, // green
			0x00, 0x00, 0xFF, 0xFF, // blue
			0xFF, 0xFF, 0xFF, 0xFF, // white
		];

		TextureData {
			width: 2,
			height: 2,
			texel: Texel::Rgba8Srgb,
			levels: colby_asset::texture::build_chain(2, 2, base, Texel::Rgba8Srgb)
				.expect("the chain builds"),
		}
	}

	#[test]
	fn a_texture_reaches_the_screen_the_way_up_its_coordinates_say() {
		// square, so that a quarter of the frame is the same distance in world
		// units across as it is down and the arithmetic below has one number
		// in it rather than two.
		let Some(mut capture) = (match Capture::new(SQUARE, SQUARE) {
			| Ok(capture) => capture,
			| Err(error) => panic!("building the capture failed: {error}"),
		}) else {
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
		// puts each sample below on the middle of a texel. Off-center would
		// blend with the neighbor the sampler wraps around to, and the test
		// would be measuring the filter rather than the coordinates.
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
			"the same color made of two things looks like two things: {dielectric:?} against 			 {metallic:?}"
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
	fn green_across_the_middle(image: &Image) -> bool {
		let middle = SIZE.1 / 2;

		(middle - 6..=middle + 6).any(|y| {
			let pixel = image.pixel(SIZE.0 / 2, y);

			dominant(pixel) == 1 && pixel[1] > 100
		})
	}

	/// A world holding one debug segment along x, at a given depth.
	fn with_a_line(at_z: f32, on_top: bool) -> World {
		let mut world = looking_world();
		world.ambient = Vec3::splat(1.0);

		let (from, to) = (Vec3::new(-3.0, 0.0, at_z), Vec3::new(3.0, 0.0, at_z));
		let green = rgb(0.1, 0.9, 0.1);

		if on_top {
			world.debug.on_top().line(from, to, green);
		} else {
			world.debug.line(from, to, green);
		}

		world
	}

	#[test]
	fn a_debug_segment_reaches_the_screen() {
		let Some(mut capture) = capture() else {
			return;
		};

		let mut world = with_a_line(0.0, false);
		let image = capture
			.shoot(&mut world)
			.expect("the capture renders");

		assert!(
			green_across_the_middle(&image),
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
		let mut world = with_a_line(-2.0, false);
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

		assert!(
			!green_across_the_middle(&image),
			"the whole point of drawing inside the scene's pass is that the depth buffer applies"
		);
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
		let mut world = with_a_line(-2.0, true);
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

		assert!(
			green_across_the_middle(&image),
			"a contact normal starts on the surface that made it, so half of the debug drawing \
			 would be invisible without this"
		);
	}

	/// How bright a pixel is, as the sum of its three color channels.
	fn brightness(pixel: [u8; 4]) -> u32 {
		u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2])
	}

	/// A two-texel normal map: the left texel leans towards `-x`, the right one
	/// towards `+x`, both by about forty-five degrees.
	///
	/// One level and no chain, so that what reaches the sampler is what is
	/// written here rather than an average of it.
	fn leaning_normals() -> TextureData {
		// tangent space, and the quad's tangent runs along +x. 37 and 217 are
		// -0.707 and +0.707 folded into a byte; 128 is zero.
		let base = vec![37, 128, 217, 255, 217, 128, 217, 255];

		TextureData {
			width: 2,
			height: 1,
			texel: Texel::Rgba8Unorm,
			levels: vec![base],
		}
	}

	#[test]
	fn a_normal_map_turns_the_light_a_surface_catches() {
		// square, for the reason the texture test is: the quad then fills the
		// frame in both directions, and a sample a quarter of the way across
		// lands on the middle of a texel rather than in the blend between two.
		let Some(mut capture) = (match Capture::new(SQUARE, SQUARE) {
			| Ok(capture) => capture,
			| Err(error) => panic!("building the capture failed: {error}"),
		}) else {
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

	/// Two texels side by side, red then blue, with no chain under them.
	fn halves() -> TextureData {
		TextureData {
			width: 2,
			height: 1,
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
		let mut capture = match Capture::new(SQUARE, SQUARE) {
			| Ok(capture) => capture?,
			| Err(error) => panic!("building the capture failed: {error}"),
		};

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
}
