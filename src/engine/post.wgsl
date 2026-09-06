// Everything that happens to a picture after the world has been drawn into it.
//
// The world is drawn into a sixteen-bit float target, where a value past one is
// a real number rather than white, and this file is what squeezes that down
// onto a screen. Four fragment entry points over one vertex entry point, and
// they are four passes rather than one because three of them read what the one
// before them wrote.
//
// **Its own file rather than a corner of `shader.wgsl`.** The default limit on
// how many bind groups a pipeline may declare is four, and the scene already
// uses nought through three - so a composite living in that module would have
// to share the material's group one with a texture that is not a material's.
// The cost is that this file is baked in rather than watched, the way
// `lines.wgsl` and `shadow.wgsl` already are.

// x is which curve, as ToneMap in declaration order; y is the reinhard white
// point; z is the exposure to use when it is not measured; w is whether it is.
// x is the key the meter aims for, which is middle grey moved by the bias in
// stops; y and z are the smallest and largest exposure a measurement may ask
// for; w is how far towards a new measurement this frame moves the eye.
struct Tuning {
    curve: vec4<f32>,
    meter: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

@group(1) @binding(0) var source: texture_2d<f32>;
@group(1) @binding(1) var source_sampler: sampler;

// The eye as it was at the end of the last frame, one texel of log luminance.
// Bound by the two passes that have an opinion about it and by nothing else.
@group(2) @binding(0) var eye: texture_2d<f32>;

struct ScreenOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle covering the screen, out of nothing but the vertex index.
//
// Three vertices rather than a quad's four: a quad has a seam down the middle
// that the rasterizer samples twice, and this has no vertex buffer to bind.
@vertex
fn vertex_screen(@builtin(vertex_index) index: u32) -> ScreenOutput {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;

    var output: ScreenOutput;
    output.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    // clip space counts y upwards and a texture counts it down.
    output.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);

    return output;
}

// How bright a color is to an eye, in the usual weights.
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The dimmest average a measurement is believed at.
//
// Matched by `colby_core::abi::post::DARKEST`. A frame of pure black would ask
// for an infinite exposure, and an infinity that reaches a uniform is a picture
// of nothing.
const DARKEST: f32 = 1.0e-4;

// The picture reduced to a small square of log luminance.
//
// Four taps rather than one: the target is a fixed hundred and twenty-eight
// square whatever the window is, so one texel of it covers a block of the
// picture, and a single sample would be a meter reading one pixel out of every
// eighty. The offsets are a quarter of a texel, so with a linear sampler the
// four together average a two-by-two block of the *output*, which is sixteen
// samples of the input for four fetches.
//
// The log is what makes the average a geometric mean, which is what a meter
// wants: one white pixel in a dark room should not open the eye all the way.
@fragment
fn fragment_luminance(input: ScreenOutput) -> @location(0) vec4<f32> {
    let step = 0.25 / vec2<f32>(textureDimensions(source, 0));
    var total = 0.0;

    for (var y = -1; y <= 1; y += 2) {
        for (var x = -1; x <= 1; x += 2) {
            let at = input.uv + vec2<f32>(f32(x), f32(y)) * step;
            let bright = luminance(textureSample(source, source_sampler, at).rgb);

            total += log2(max(bright, DARKEST));
        }
    }

    return vec4<f32>(total * 0.25, 0.0, 0.0, 1.0);
}

// The same picture at half the width and half the height.
//
// One tap, because the sampler is linear and the target is exactly half the
// source: a single fetch at the middle of a two-by-two block *is* its average.
// Run until the target is one texel, at which point it holds the whole
// picture's log-average.
@fragment
fn fragment_halve(input: ScreenOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSample(source, source_sampler, input.uv).r, 0.0, 0.0, 1.0);
}

// Where the eye has got to, one texel.
//
// **A frame with no history arrives already adapted.** `tuning.meter.w` is one
// when nothing has been measured yet, which makes this the measurement rather
// than a step towards it - and that is not a convenience. A process that
// renders exactly one frame is what `--shot` is, and a picture of a half-open
// eye would be both wrong and different from the same scene in a window that
// had been up for a second.
//
// The blend is in log space, because that is the space the number is in and
// because a stop is a stop whichever end of the range it is at.
@fragment
fn fragment_adapt(input: ScreenOutput) -> @location(0) vec4<f32> {
    let measured = textureSample(source, source_sampler, input.uv).r;
    let before = textureLoad(eye, vec2<i32>(0, 0), 0).r;

    return vec4<f32>(mix(before, measured, tuning.meter.w), 0.0, 0.0, 1.0);
}

// `c / (1 + c)`, scaled so that the white point reaches one.
//
// The cheapest curve that is a curve. Nothing desaturates and no hue shifts,
// which is what makes it read flat beside the one below.
fn reinhard(color: vec3<f32>, white: f32) -> vec3<f32> {
    let w = max(white, 0.0001);

    return color * (1.0 + color / (w * w)) / (1.0 + color);
}

// The fitted ACES curve, Narkowicz's approximation.
//
// Six constants and one divide standing in for a transform the film industry
// ships as a pair of lookup tables. It has a look - contrast through the
// middle, a long shoulder, brights sliding towards white - and the look is why
// it is the default rather than the cheaper one above.
fn aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// The picture, exposed and squeezed, into whatever the screen's format is.
//
// The exposure is one line and it is the same line
// `colby_core::abi::post::Post::metered` is: the key it divides by has already
// had the bias in stops folded into it on the way to the uniform, so what is
// left here is the divide and the clamp. Nothing is read back to the processor
// to do it - a stall to fetch one float would cost more than the whole chain
// that produced it.
@fragment
fn fragment_composite(input: ScreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source, source_sampler, input.uv).rgb;
    var exposure = tuning.curve.z;

    if (tuning.curve.w > 0.5) {
        let average = exp2(textureLoad(eye, vec2<i32>(0, 0), 0).r);

        exposure = clamp(tuning.meter.x / max(average, DARKEST), tuning.meter.y, tuning.meter.z);
    }

    let exposed = color * exposure;
    let curve = i32(tuning.curve.x);

    if (curve == 2) {
        return vec4<f32>(aces(exposed), 1.0);
    }

    if (curve == 1) {
        return vec4<f32>(clamp(reinhard(exposed, tuning.curve.y), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    // and `none`, which is what the eight-bit target did on its own before
    // there was a curve at all. The exposure still applies, which is what
    // Godot's `LINEAR` means too.
    return vec4<f32>(clamp(exposed, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
