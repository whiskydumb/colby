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
			Decal, Emitter, Light, MeshId, Renderable, Spark, SparkBlend, Texel, TextureData,
			TextureId, ToneMap, Transform, Value, World,
			cvar::Cvars,
			debug,
			material::{Blend, Material},
		},
		glam::{Quat, Vec3},
	};
	use wgpu::TextureFormat;

	use crate::{Capture, Gpu, depth, occlusion, prepass, scene, shadow};

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

		// a decal over the first of them, throwing a picture and a normal map, so
		// the atlas is built, written and bound through both of its views; and
		// the second of them refusing decals, so the flag reaches the shader
		let picture = world.textures.insert("test/splash", TextureData {
			width: 4,
			height: 4,
			faces: 1,
			texel: Texel::Rgba8Srgb,
			levels: vec![vec![0xFF; 64]],
		});
		let bumps = world
			.textures
			.insert("test/splash_normal", TextureData {
				width: 4,
				height: 4,
				faces: 1,
				texel: Texel::Rgba8Unorm,
				levels: vec![[128, 128, 255, 255].repeat(16)],
			});
		let splash = world
			.materials
			.insert("test/splash", Material::textured(picture).bumped(bumps));
		let decal = world.entities.spawn_at(Transform {
			position: Vec3::ZERO,
			rotation: Quat::IDENTITY,
			scale: Vec3::splat(2.0),
		});

		world
			.entities
			.set_renderable(decal, Renderable { material: splash, ..Renderable::NOTHING });
		world.entities.set_decal(decal, Decal::BOX);

		let second = world.entities.iter().nth(1).map(|(id, ..)| id);
		if let Some(second) = second {
			world.entities.set_takes_decals(second, false);
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

		// a point and a cone, both casting, so the atlas's local layer is
		// drawn into and both the six-tile shape and the one-tile shape are
		// recorded. @ref [`shadow`](crate::shadow).
		let point = world
			.entities
			.spawn_at(Transform::at(Vec3::new(0.0, 2.0, 1.0)));
		world
			.entities
			.set_light(point, Light::point(Vec3::ONE, 8.0, 12.0));

		let spot = world
			.entities
			.spawn_at(Transform::at(Vec3::new(2.0, 2.0, 1.0)));
		world
			.entities
			.set_light(spot, Light::spot(Vec3::ONE, 8.0, 12.0, 0.2, 0.6));

		world
			.debug
			.line(Vec3::ZERO, Vec3::X, debug::WHITE);

		world
	}

	/// The two variables a scene reads off the world when it draws.
	fn tuned(world: &mut World, samples: &str) {
		let mut cvars = Cvars::new();

		cvars.var(scene::MSAA, Value::Float(1.0), "");
		cvars.var(shadow::LOCAL_LAMPS, Value::Float(shadow::DEFAULT_LOCAL_LAMPS), "");
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
	fn and_draws_the_depth_instead_at_one_sample_and_at_four() {
		// the view's pipeline and its bind group, and at four samples the pass
		// that resolves the depth, are only recorded in a frame that asks for
		// the depth; the two tests above never do
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		for samples in ["1", "4", "1"] {
			tuned(&mut world, samples);
			world
				.cvars
				.var(depth::VIEW, Value::Float(depth::NO_VIEW), "");
			world.cvars.set(depth::VIEW, "10");

			capture
				.shoot(&mut world)
				.expect("the depth drawn instead of the picture renders");
		}
	}

	#[test]
	fn and_writes_every_surface_before_the_scene_at_one_sample_and_at_four() {
		// the pass before the scene, its four pipelines and the view that draws
		// its buffer, all of them only recorded in a frame that asks - so the
		// three tests above never build any of them. Both of the view's two
		// answers, across both sample counts, and back.
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		for (samples, showing) in [("1", "1"), ("4", "2"), ("1", "2"), ("4", "1")] {
			tuned(&mut world, samples);
			world
				.cvars
				.var(prepass::VIEW, Value::Float(prepass::NO_VIEW), "");
			world.cvars.set(prepass::VIEW, showing);

			capture
				.shoot(&mut world)
				.expect("what every surface is, drawn instead of the picture, renders");
		}
	}

	#[test]
	fn and_works_out_how_much_of_the_sky_every_pixel_sees_at_one_sample_and_at_four() {
		// the estimate and the average are only recorded in a frame something
		// asks for them in, and both read what the pass before the scene wrote -
		// so this builds and validates the first two readers of that buffer and
		// the view that draws what they work out, across both sample counts
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		for samples in ["1", "4", "1"] {
			tuned(&mut world, samples);
			world
				.cvars
				.var(prepass::VIEW, Value::Float(prepass::NO_VIEW), "");
			world.cvars.set(prepass::VIEW, "3");

			capture.shoot(&mut world).expect(
				"how much of the sky each pixel sees, drawn instead of the picture, renders",
			);
		}
	}

	#[test]
	fn and_multiplies_the_share_into_the_picture_and_binds_one_texel_when_nothing_asks() {
		// the picture asking on its own, with no view: group nought made over the
		// half-sized buffer, then over the one texel that says all of the sky when
		// the strength goes to nought, and back - across both sample counts, so
		// that every one of the scene's pipelines is bound against both
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		for (samples, strength) in [("1", "1"), ("4", "0"), ("4", "1"), ("1", "0"), ("1", "0.5")]
		{
			tuned(&mut world, samples);
			world
				.cvars
				.var(occlusion::STRENGTH, Value::Float(occlusion::DEFAULT_STRENGTH), "");
			world.cvars.set(occlusion::STRENGTH, strength);

			capture
				.shoot(&mut world)
				.expect("a picture that takes the share of the sky away renders");
		}
	}

	#[test]
	fn and_smears_the_sky_around_the_sun_at_one_sample_and_at_four() {
		// the three shaft passes are only recorded in a frame that asks for
		// them, and the mask reads the depth the way the view above does - so
		// this is the second reader of that buffer, built and validated here
		// on a device that draws nothing
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		// the camera looks down -z from +z, and light traveling +z comes from
		// -z, so this is the sun in front of it
		world.light = Vec3::Z;
		world.post.shafts = 0.8;

		for samples in ["1", "4", "1"] {
			tuned(&mut world, samples);
			capture
				.shoot(&mut world)
				.expect("the smear renders at either sample count");
		}
	}

	#[test]
	fn and_blurs_what_the_lens_is_not_focused_on_at_one_sample_and_at_four() {
		// the three lens passes are only recorded in a frame whose camera
		// focuses on something, and two of them read the depth - so this is
		// the third reader of that buffer, built and validated here on a
		// device that draws nothing. The last of the three blends into the
		// picture, which is a pipeline state the two above do not have.
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		world.camera.focus = 4.0;
		world.camera.focus_range = 6.0;
		world.camera.blur = 12.0;

		for samples in ["1", "4", "1"] {
			tuned(&mut world, samples);
			capture
				.shoot(&mut world)
				.expect("the blur renders at either sample count");
		}
	}

	#[test]
	fn and_lights_the_air_with_every_lamp_at_one_sample_and_at_four() {
		// the march and the pass that puts it over the picture are built from the
		// scene's own source the first frame a world's air is hazy in, and a pass
		// that would not build only says so and leaves the air clear - so this
		// asks whether they were built, and lets the average and the view of the
		// air be validated beside them, across both sample counts and back
		let Some(gpu) = headless() else {
			return;
		};
		let mut capture = Capture::new(&gpu, SIZE.0, SIZE.1).expect("the capture builds");
		let mut world = everything();

		world.post.haze = 0.05;

		for (samples, showing) in [("1", "0"), ("4", "7"), ("1", "7"), ("4", "0")] {
			tuned(&mut world, samples);
			world
				.cvars
				.var(prepass::VIEW, Value::Float(prepass::NO_VIEW), "");
			world.cvars.set(prepass::VIEW, showing);

			capture
				.shoot(&mut world)
				.expect("the air renders at either sample count");

			assert!(
				capture.scene_mut().haze_built(),
				"and its passes were built and validated rather than left out with a warning"
			);
		}
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
