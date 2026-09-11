//! The specular lobe, run on the device and held against its arithmetic.
//!
//! The two functions the surface shader lights with are cut out of
//! `shader.wgsl` as they stand and run in a compute pass, one answer a
//! question, so what is checked is the text the scene compiles rather than a
//! copy of it in another language. A picture cannot say this: the peak of a
//! smooth highlight is thousands of times brighter than white, and eight bits
//! of it say only that it is white.

use std::f64::consts::PI;

use colby_core::{abi::material::MIN_ROUGHNESS, bytemuck};
use wgpu::{
	BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages,
	CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, MapMode,
	PipelineCompilationOptions, PollType, ShaderModuleDescriptor, ShaderSource,
};

use crate::gpu::{self, Gpu};

/// The surface shader, as the scene compiles it.
const SOURCE: &str = include_str!("shader.wgsl");

/// What comes before the expression in the pass that asks: four numbers a
/// question, one thread each, and one number back.
const HEAD: &str = "
@group(0) @binding(0) var<storage, read> asked: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> answered: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&asked)) {
        return;
    }

    let it = asked[id.x];
    answered[id.x] = ";

/// What comes after it.
const TAIL: &str = ";
}
";

/// Roughnesses from the smoothest drawn to chalk, thickest where the floor the
/// divisor used to have flattened the highlight.
const ROUGHNESSES: [f32; 12] =
	[MIN_ROUGHNESS, 0.05, 0.06, 0.08, 0.1, 0.16, 0.2, 0.25, 0.27, 0.3, 0.5, 1.0];

/// How many steps the lobe's integral is taken in.
const STEPS: u16 = 2048;

/// One of the shader's functions, from its `fn` to the brace that closes it.
///
/// @param name - the function
/// @return its whole text
fn function(name: &str) -> String {
	let opening = format!("\nfn {name}(");
	let (_, rest) = SOURCE
		.split_once(opening.as_str())
		.unwrap_or_else(|| panic!("shader.wgsl has no function called {name}"));
	let (body, _) = rest
		.split_once("\n}")
		.unwrap_or_else(|| panic!("{name} in shader.wgsl never closes"));

	format!("fn {name}({body}\n}}\n")
}

/// Asks the device one expression a question.
///
/// @param gpu - the device
/// @param functions - the shader's own functions the expression calls
/// @param expression - WGSL over `it`, the question's four numbers
/// @param questions - what to ask
/// @return one answer a question, in the order asked
fn answers(
	gpu: &Gpu,
	functions: &[String],
	expression: &str,
	questions: &[[f32; 4]],
) -> Vec<f32> {
	let device = gpu.device();
	let source = [functions.concat().as_str(), HEAD, expression, TAIL].concat();
	let module = device.create_shader_module(ShaderModuleDescriptor {
		label: Some("lobe"),
		source: ShaderSource::Wgsl(source.into()),
	});
	let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
		label: Some("lobe"),
		layout: None,
		module: &module,
		entry_point: Some("main"),
		compilation_options: PipelineCompilationOptions::default(),
		cache: None,
	});

	let count = u32::try_from(questions.len()).expect("a test asks a few thousand questions");
	let size = u64::from(count) * 4;
	let asked = device.create_buffer(&BufferDescriptor {
		label: Some("lobe questions"),
		size: size * 4,
		usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});
	let answered = device.create_buffer(&BufferDescriptor {
		label: Some("lobe answers"),
		size,
		usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
		mapped_at_creation: false,
	});
	let readback = device.create_buffer(&BufferDescriptor {
		label: Some("lobe readback"),
		size,
		usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
		mapped_at_creation: false,
	});
	gpu.queue()
		.write_buffer(&asked, 0, bytemuck::cast_slice(questions));

	let group = device.create_bind_group(&BindGroupDescriptor {
		label: Some("lobe"),
		layout: &pipeline.get_bind_group_layout(0),
		entries: &[
			BindGroupEntry {
				binding: 0,
				resource: asked.as_entire_binding(),
			},
			BindGroupEntry {
				binding: 1,
				resource: answered.as_entire_binding(),
			},
		],
	});

	let mut encoder =
		device.create_command_encoder(&CommandEncoderDescriptor { label: Some("lobe") });
	let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
		label: Some("lobe"),
		timestamp_writes: None,
	});
	pass.set_pipeline(&pipeline);
	pass.set_bind_group(0, &group, &[]);
	pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
	drop(pass);
	encoder.copy_buffer_to_buffer(&answered, 0, &readback, 0, size);
	gpu.queue().submit([encoder.finish()]);

	let slice = readback.slice(..);
	slice.map_async(MapMode::Read, |_| {});
	device
		.poll(PollType::Wait { submission_index: None, timeout: None })
		.expect("the device answers");

	let view = slice.get_mapped_range().expect("the answers map");
	let answers = bytemuck::cast_slice::<u8, f32>(&view).to_vec();

	drop(view);
	readback.unmap();

	answers
}

/// The distribution in double precision, in its textbook form, where a
/// roughness to the fourth is nowhere near the rounding.
fn exact_distribution(normal_dot_half: f64, roughness: f64) -> f64 {
	let a2 = roughness.powi(4);
	let divisor = (normal_dot_half * normal_dot_half).mul_add(a2 - 1.0, 1.0);

	a2 / (PI * divisor * divisor)
}

/// The shadowing term in double precision.
fn exact_visibility(normal_dot_view: f64, normal_dot_light: f64, roughness: f64) -> f64 {
	let a2 = roughness.powi(4);
	let view = normal_dot_light
		* (normal_dot_view * normal_dot_view)
			.mul_add(1.0 - a2, a2)
			.sqrt();
	let light = normal_dot_view
		* (normal_dot_light * normal_dot_light)
			.mul_add(1.0 - a2, a2)
			.sqrt();

	0.5 / (view + light)
}

#[test]
fn a_highlight_peaks_at_one_over_pi_a_squared_however_smooth_the_surface() {
	// the peak is where the half vector is the normal, and there the
	// distribution is `1 / (pi a^2)` for every roughness; the floor the divisor
	// used to have made it 1 at 0.1 and 0.041 at the smoothest
	let Some(gpu) = gpu::shared() else {
		return;
	};

	let questions = ROUGHNESSES.map(|roughness| [1.0, roughness, 0.0, 0.0]);
	let peaks =
		answers(gpu, &[function("distribution_ggx")], "distribution_ggx(it.x, it.y)", &questions);

	for (roughness, peak) in ROUGHNESSES.into_iter().zip(peaks) {
		let exact = exact_distribution(1.0, f64::from(roughness));
		let off = (f64::from(peak) - exact).abs() / exact;

		assert!(
			off < 1.0e-5,
			"at a roughness of {roughness} the peak is {peak}, where the arithmetic says \
			 {exact}: {off:e} out"
		);
	}
}

#[test]
fn the_lobe_falls_away_from_its_peak_the_way_the_arithmetic_does() {
	// out from the peak in steps of the lobe's own width and on to where the
	// half vector lies in the surface. The one rounding the shader cannot
	// avoid is in `1 - n.h^2`, whose `n.h^2` a float near one holds to a step
	// of about six hundred-millionths, so that is the tolerance: tight at a
	// middling roughness and a few percent right beside the smoothest peak
	let Some(gpu) = gpu::shared() else {
		return;
	};

	let mut questions = Vec::new();
	for roughness in [MIN_ROUGHNESS, 0.1, 0.3, 0.8] {
		let width = roughness * roughness;
		for angle in [0.25 * width, 0.5 * width, width, 2.0 * width, 4.0 * width, 16.0 * width]
			.into_iter()
			.chain([0.3, 1.0, 1.5])
		{
			questions.push([angle.cos(), roughness, 0.0, 0.0]);
		}
	}

	let lobe =
		answers(gpu, &[function("distribution_ggx")], "distribution_ggx(it.x, it.y)", &questions);

	for (question, answer) in questions.iter().zip(lobe) {
		let normal_dot_half = f64::from(question[0]);
		let roughness = f64::from(question[1]);
		let exact = exact_distribution(normal_dot_half, roughness);
		let divisor = (normal_dot_half * normal_dot_half).mul_add(roughness.powi(4) - 1.0, 1.0);
		let within = 2.0_f64.mul_add(2.0_f64.powi(-24) / divisor, 1.0e-5);
		let off = (f64::from(answer) - exact).abs() / exact;

		assert!(
			off < within,
			"at a roughness of {roughness} and n.h {normal_dot_half} the lobe is {answer}, \
			 where the arithmetic says {exact}: {off:e} out, against {within:e}"
		);
	}
}

#[test]
fn the_lobe_holds_all_of_its_light_at_every_roughness() {
	// the integral of D times n.h over the hemisphere is one for any
	// microfacet distribution, whatever its shape; the floor under the old
	// divisor kept three and a half percent of it at 0.1 and half at 0.2. The
	// steps are even in the logarithm of `1 - n.h`, from a hair off the peak
	// to a quarter turn, and the sum is over the n.h the device was handed
	let Some(gpu) = gpu::shared() else {
		return;
	};

	let roughnesses = [MIN_ROUGHNESS, 0.1, 0.2, 0.5, 1.0];
	let mut questions = Vec::new();
	for roughness in roughnesses {
		for step in 0..=STEPS {
			let away = 10.0_f32.powf(12.0_f32.mul_add(f32::from(step) / f32::from(STEPS), -12.0));
			questions.push([1.0 - away, roughness, 0.0, 0.0]);
		}
	}

	let lobe =
		answers(gpu, &[function("distribution_ggx")], "distribution_ggx(it.x, it.y)", &questions);
	let steps = usize::from(STEPS) + 1;

	for (index, roughness) in roughnesses.into_iter().enumerate() {
		let asked = questions
			.get(index * steps..(index + 1) * steps)
			.expect("one run a roughness");
		let got = lobe
			.get(index * steps..(index + 1) * steps)
			.expect("and its answers");
		let whole = held(asked, got);

		assert!(
			(whole - 1.0).abs() < 2.0e-3,
			"at a roughness of {roughness} the lobe holds {whole} of its light, where every \
			 distribution holds one"
		);
	}
}

/// The integral of a lobe times n.h over the hemisphere, by trapezoids.
///
/// @param asked - the questions, `1 - n.h` growing
/// @param got - the lobe at each
/// @return two pi times the integral over `1 - n.h` of the lobe times n.h
fn held(asked: &[[f32; 4]], got: &[f32]) -> f64 {
	let mut total = 0.0;
	let mut last: Option<(f64, f64)> = None;

	for (question, answer) in asked.iter().zip(got) {
		let normal_dot_half = f64::from(question[0]);
		let away = 1.0 - normal_dot_half;
		let value = f64::from(*answer) * normal_dot_half;

		if let Some((before_away, before_value)) = last {
			total = (0.5 * (value + before_value)).mul_add(away - before_away, total);
		}

		last = Some((away, value));
	}

	2.0 * PI * total
}

/// Every pairing of a view and a light the shadowing test asks about, at one
/// roughness: from both head on to both grazing, the light as far as lying in
/// the surface, the eye no further than the shader lets it.
fn grazing(roughness: f32) -> Vec<[f32; 4]> {
	let mut asked = Vec::new();

	for view in [1.0e-4, 1.0e-3, 0.01, 0.1, 0.5, 1.0] {
		for light in [0.0, 1.0e-3, 0.01, 0.1, 0.5, 1.0] {
			asked.push([view, light, roughness, 0.0]);
		}
	}

	asked
}

#[test]
fn the_shadowing_term_is_the_arithmetic_s_even_where_both_directions_graze() {
	// the eye arrives at no less than a ten-thousandth, so with the roughness
	// at its floor and the light in the surface the divisor is two
	// ten-millionths and the term two and a half million, which a float holds;
	// the floor the divisor used to have made it five thousand there
	let Some(gpu) = gpu::shared() else {
		return;
	};

	let questions: Vec<[f32; 4]> = [MIN_ROUGHNESS, 0.3, 1.0]
		.into_iter()
		.flat_map(grazing)
		.collect();
	let terms = answers(
		gpu,
		&[function("visibility_smith")],
		"visibility_smith(it.x, it.y, it.z)",
		&questions,
	);

	for (question, term) in questions.iter().zip(terms) {
		let (view, light, roughness) =
			(f64::from(question[0]), f64::from(question[1]), f64::from(question[2]));
		let exact = exact_visibility(view, light, roughness);
		let off = (f64::from(term) - exact).abs() / exact;

		assert!(
			term.is_finite() && off < 1.0e-5,
			"at n.v {view}, n.l {light} and a roughness of {roughness} the term is {term}, \
			 where the arithmetic says {exact}: {off:e} out"
		);
	}
}

#[test]
fn the_shader_and_the_material_agree_on_the_smoothest_surface_drawn() {
	// a module cannot read a constant out of Rust, so the number is written
	// twice and this is what stops the two drifting apart
	let declaration = "const MIN_ROUGHNESS: f32 = ";
	let value = SOURCE
		.split_once(declaration)
		.and_then(|(_, rest)| rest.split_once(';'))
		.map(|(value, _)| value.trim())
		.unwrap_or_else(|| panic!("shader.wgsl declares no {declaration}"));

	assert_eq!(
		value.parse::<f32>().ok(),
		Some(MIN_ROUGHNESS),
		"the shader holds a roughness to the number the material says it does"
	);
}

/// The golden angle, which spreads points evenly round a sphere.
const GOLDEN_ANGLE: f32 = 2.399_963;

/// Directions spread over the sphere, each with the light and the eye a third
/// of a radian either side of it.
///
/// @param count - how many
/// @return one question a direction: the direction, then the angle
fn mirrored_around(count: u16) -> Vec<[f32; 4]> {
	let mut asked = Vec::new();

	for index in 0..count {
		let place = f32::from(index) + 0.5;
		let height = 2.0_f32.mul_add(-place / f32::from(count), 1.0);
		let across = height.mul_add(-height, 1.0).max(0.0).sqrt();
		let turn = GOLDEN_ANGLE * place;

		asked.push([across * turn.cos(), height, across * turn.sin(), 0.3]);
	}

	asked
}

#[test]
fn the_highlight_never_climbs_past_its_own_peak_where_the_light_and_the_eye_mirror() {
	// a light and an eye the same angle either side of the normal put the half
	// vector on the normal, and there `lit_by` of a white metal is the peak of
	// the lobe times the rest of the term, exactly. Two unit vectors made that
	// way dot to a hair past one for about one direction in forty on the RX
	// this was written on (110 of 4096), and without `normal_dot_half` held
	// under one those directions climbed a tenth past the peak
	let Some(gpu) = gpu::shared() else {
		return;
	};

	let helper = format!(
		"fn mirrored(it: vec4<f32>) -> f32 {{
    let normal = normalize(it.xyz);
    let across = normalize(cross(normal, vec3<f32>(0.3, 0.9, 0.2)));
    let towards_light = normalize(normal * cos(it.w) + across * sin(it.w));
    let towards_eye = normalize(normal * cos(it.w) - across * sin(it.w));
    let normal_dot_view = max(dot(normal, towards_eye), 0.0001);

    return lit_by(normal, towards_eye, towards_light, vec3<f32>(1.0), vec3<f32>(0.0), \
		 {MIN_ROUGHNESS:?}, normal_dot_view).x;
}}
"
	);

	let questions = mirrored_around(4096);
	let terms = answers(
		gpu,
		&[
			function("fresnel_schlick"),
			function("distribution_ggx"),
			function("visibility_smith"),
			function("lit_by"),
			helper,
		],
		"mirrored(it)",
		&questions,
	);

	let a = f64::from(MIN_ROUGHNESS).powi(2);
	let mut highest = 0.0_f64;
	for (question, term) in questions.iter().zip(terms) {
		let facing = f64::from(question[3]).cos();
		let peak = exact_visibility(facing, facing, f64::from(MIN_ROUGHNESS)) * facing / (a * a);

		highest = highest.max(f64::from(term) / peak);
	}

	assert!(
		highest > 0.999 && highest < 1.0 + 1.0e-4,
		"the highlight reaches its peak and never climbs past it: {highest} of the peak at most"
	);
}
