//! Building every pipeline on a device with no graphics hardware behind it.
//!
//! The renderer's other tests draw something and read the pixels back, so
//! every one of them skips itself on a machine with no adapter - which means
//! a build with no GPU reports a green suite that checked none of the seven
//! shader stages in this crate. That is the hole this fills.
//!
//! [`Gpu::headless`](crate::Gpu::headless) opens wgpu's stub backend, on which
//! every operation is an empty body except making and mapping a buffer. What
//! still runs on it is all of wgpu's validation and all of naga's, so a shader
//! that does not compile and a pipeline that disagrees with its target are
//! both refused here exactly as they would be on a card. What does not run is
//! the drawing: **nothing in this file may look at a pixel**, because every
//! pixel it could look at is zero.
//!
//! @ref `colby-verification-loop` for where this sits: it is the cheapest of
//! the four stages and the only one that needs no hardware, and it does not
//! replace a single one of the pixel tests.

#[cfg(test)]
mod tests {
	use colby_core::{
		abi::{
			Emitter, MeshId, Renderable, Spark, SparkBlend, TextureId, ToneMap, Transform, Value,
			World,
			cvar::Cvars,
			debug,
			material::{Blend, Material},
		},
		glam::{Quat, Vec3},
	};
	use wgpu::TextureFormat;

	use crate::{Capture, Gpu, scene};

	/// How big the target is. Small on purpose: nothing here reads it.
	const SIZE: (u32, u32) = (16, 16);

	/// A device that validates and draws nothing, or `None` without the
	/// feature.
	fn headless() -> Option<Gpu> {
		match Gpu::headless() {
			| Ok(gpu) => gpu,
			| Err(error) => panic!("opening the stub device failed: {error}"),
		}
	}

	/// A world with one of everything the renderer has a pipeline for.
	///
	/// One opaque thing, one masked, one blended, one skinned, one emitter of
	/// each blend with a particle apiece, a line for the debug pen and a
	/// tonemap that is not the identity - so that a frame drawn over this
	/// touches every entry of every table rather than the first row of one.
	fn everything() -> World {
		let mut world = World::new();

		world.post.tonemap = ToneMap::Aces;
		world.post.bloom = 0.5;
		world.camera.position = Vec3::new(0.0, 0.0, 5.0);
		world.camera.target = Vec3::ZERO;

		for (number, blend) in [Blend::Opaque, Blend::Mask, Blend::Alpha]
			.into_iter()
			.enumerate()
		{
			let across = f32::from(u8::try_from(number).unwrap_or(0));
			let material = world
				.materials
				.insert(&format!("test/{blend:?}"), Material { blend, ..Material::DEFAULT });
			let id = world.entities.spawn_at(Transform {
				position: Vec3::new(across, 0.0, 0.0),
				rotation: Quat::IDENTITY,
				scale: Vec3::ONE,
			});

			world
				.entities
				.set_renderable(id, Renderable::of(MeshId::CUBE, material, Vec3::splat(0.5)));
		}

		for blend in [SparkBlend::Alpha, SparkBlend::Additive] {
			let id = world.entities.spawn();

			world.entities.set_emitter(id, Emitter {
				blend,
				texture: TextureId::NONE,
				..Emitter::point(10.0, 1.0)
			});
			world.sparks.push(Spark {
				position: Vec3::ZERO,
				velocity: Vec3::ZERO,
				age: 0.5,
				life: 1.0,
				owner: id,
			});
		}

		world
			.debug
			.line(Vec3::ZERO, Vec3::X, debug::WHITE);

		world
	}

	/// The two variables a scene reads off the world when it draws.
	fn tuned(world: &mut World, samples: &str) {
		let mut cvars = Cvars::new();

		cvars.var(scene::MSAA, Value::Float(1.0), "");
		cvars.set(scene::MSAA, samples);
		world.cvars = cvars;
	}

	#[test]
	fn every_pipeline_the_renderer_has_builds_on_a_device_with_no_gpu() {
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		tuned(&mut world, "1");

		capture
			.shoot(&mut world)
			.expect("a frame with one of everything in it renders");

		assert_eq!(capture.scene_mut().sparks(), 2, "both particles were laid out");
		assert!(
			capture.scene_mut().sparks_built(),
			"and the two pipelines they need were built - which the count above does not say, 			 because it is filled before anything is asked to compile"
		);
	}

	#[test]
	fn and_builds_again_at_four_samples_and_in_the_windows_format() {
		// the two axes that make a second whole table of pipelines: the sample
		// count is baked into every one of them, and so is the color format
		// the fragment stage writes. A shader edit that is fine at one sample
		// in one format and not at four in the other is a real shape of bug -
		// `r.msaa` is a live variable, and a screenshot taken beside a window
		// is drawn in the window's format.
		let Some(gpu) = headless() else {
			return;
		};
		let mut window = Capture::in_format(&gpu, TextureFormat::Bgra8UnormSrgb, SIZE.0, SIZE.1)
			.expect("the window's format builds");
		let mut world = everything();

		tuned(&mut world, "4");

		window
			.shoot(&mut world)
			.expect("four samples in the window's format render");
	}

	#[test]
	fn a_shader_that_does_not_compile_is_refused_here_the_way_a_card_refuses_it() {
		// what says the two above have teeth. Nothing is drawn on this device,
		// so "the frame rendered" could mean the pipelines were never built -
		// this is the negative control that says they were, and that naga ran.
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let source = include_str!("shader.wgsl").replace("fn vertex_main", "fn vertex_broken");

		assert!(
			capture.scene_mut().set_shader(&source).is_err(),
			"a shader with no entry point by that name does not compile, here as anywhere"
		);
	}

	#[test]
	fn the_stub_draws_nothing_at_all_which_is_why_no_test_here_reads_a_pixel() {
		// the standing warning, as an assertion. If wgpu ever teaches this
		// backend to copy a texture, this test is where that turns up - and
		// the note in every file that says "no pixel" can be revisited.
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = World::new();

		world.clear = Vec3::new(1.0, 0.0, 0.0);

		let image = capture
			.shoot(&mut world)
			.expect("a frame of nothing renders");

		assert!(
			image.pixels.iter().all(|byte| *byte == 0),
			"a red clear comes back black, because nothing here clears anything"
		);
	}
}
