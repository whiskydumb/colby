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
// x is how much of the bright pass is added back; y is how bright a pixel has
// to be to be in it; z is the width of the knee under that; w is unused.
// x is how far away white is while the depth is drawn instead of the picture;
// y and z are the two numbers of the projection a stored depth is turned back
// into a distance with, its `z_axis.z` and its `w_axis.z`; w is whether the
// target applies the sRGB curve on the way out. While what the pass before the
// scene wrote is drawn instead, x is one for the normal and two for the
// roughness and w is the same curve. All nought otherwise.
struct Tuning {
    curve: vec4<f32>,
    meter: vec4<f32>,
    bloom: vec4<f32>,
    depth: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

@group(1) @binding(0) var source: texture_2d<f32>;
@group(1) @binding(1) var source_sampler: sampler;

// The eye as it was at the end of the last frame, one texel of log luminance.
// Bound by the two passes that have an opinion about it and by nothing else.
@group(2) @binding(0) var eye: texture_2d<f32>;

// The widest rung of the bloom chain, which is everything the chain gathered.
// Bound by the composite alone, and only when there is any bloom to add.
@group(3) @binding(0) var bloom: texture_2d<f32>;
@group(3) @binding(1) var bloom_sampler: sampler;

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

// How many taps a reduction takes across each axis: four, so sixteen in all.
//
// Matched by `METER_STEP` in `post.rs`, and the two have to agree: with a
// linear sampler every tap is itself the average of a two-by-two block, so
// sixteen taps stand for exactly an eight-by-eight one. A step of anything but
// eight makes the reduction sixteen evenly spread samples of the block rather
// than all of it.
//
// **Nothing to do with [`METER_TAPS`]**, which is how many the *first* pass
// takes off the picture. A reduction is exact and wants a regular grid; the
// first pass is a sample of something much larger than itself and wants an
// irregular one.
const TAP_ROWS: i32 = 4;

// Where one tap sits, as a share of what an output texel covers.
//
// A four-by-four grid centered on the texel: at plus and minus an eighth and
// three eighths, the sixteen of them tile the footprint exactly, with no gap
// and no overlap. Arithmetic rather than a table because WGSL will not index a
// module constant with a loop variable.
fn tap(at: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(at) - 1.5) * 0.25;
}

// What one output texel covers, in the input's texture coordinates.
//
// Read off the interpolator rather than passed in: uv runs from nought to one
// across the target, so its derivative across one pixel *is* one output texel,
// whatever size the target happens to be. Exact for a full-screen triangle, and
// it is why nothing here has to be told how big the thing it draws into is.
fn footprint(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(dpdx(uv).x, dpdy(uv).y);
}

// The mean of the block one output texel stands for.
//
// Shared by the reduction and by the eye, which is the whole reason the eye
// needs no pass of its own any more: the last rung is eight texels across, the
// eye is one, and that is a reduction like any other with a blend on the end.
fn reduced(uv: vec2<f32>) -> f32 {
    let step = footprint(uv);
    var total = 0.0;

    for (var y = 0; y < TAP_ROWS; y += 1) {
        for (var x = 0; x < TAP_ROWS; x += 1) {
            let at = uv + tap(vec2<i32>(x, y)) * step;

            total += textureSample(source, source_sampler, at).r;
        }
    }

    return total / f32(TAP_ROWS * TAP_ROWS);
}

// How many taps a texel of the meter takes across each axis: two, so four.
//
// **Four rather than the reduction's sixteen, and it is more accurate for it.**
// `PERF-5` asked whether sixteen was worth its nine microseconds and the answer
// turned out to be about the pattern rather than the count: sixteen taps sit
// five pixels apart at seven-twenty, and content that repeats every five pixels
// beats against them. Measured against a meter with full coverage on a striped
// wall, sixteen regular taps read the picture **seven percent** too bright and
// four jittered ones read it a tenth of a percent too bright. Fewer samples,
// placed irregularly, beat more samples in a row.
const METER_TAPS: i32 = 2;

// Where inside its own cell one tap lands, from nought to one on each axis.
//
// **The whole of what makes the meter unbiased.** Four taps in the four
// quadrants of the footprint is stratified sampling and would still be a
// regular grid; moving each one inside its quadrant by a number that depends on
// where the output texel is turns a beat with the content into noise, and noise
// over four thousand texels averages away where a beat does not.
//
// It depends on the texel and not on the frame, so the pattern is the same
// every frame and the eye does not shimmer: what this scatters is the *phase*
// of the sampling across the picture, not the answer across time.
// **And every tap it places is then snapped to a texel of the source.** @ref
// `fragment_luminance`, which does the snapping: the jitter says *which* texel
// of the block to read and never asks for a place between two. A tap that lands
// between texels is read through the linear sampler and comes back as a blend
// of both, and the log of a blend of two brightnesses is above the blend of
// their logs - so on content that alternates every pixel a continuous jitter
// reads the picture the better part of a stop too bright. Measured against
// painted pictures whose average is arithmetic: 0.83 stops out on two-pixel
// stripes before the snap and 0.18 after, worst of eight.
fn scatter(uv: vec2<f32>, cell: vec2<i32>) -> vec2<f32> {
    let seed = uv * 4096.0 + vec2<f32>(cell) * 17.0;

    return fract(
        sin(vec2<f32>(
            dot(seed, vec2<f32>(12.9898, 78.233)),
            dot(seed, vec2<f32>(39.3468, 11.135))
        )) * 43758.5453
    );
}

// The picture reduced to a small square of log luminance.
//
// Four jittered taps over the whole block an output texel stands for: the
// target is a fixed sixty-four square whatever the window is, so one texel of
// it covers about two hundred pixels of a seven-twenty picture, and a single
// sample would be a meter reading one pixel in two hundred.
//
// **This used to divide by the size of the *source*, and that was a bug.** The
// note here claimed the four taps averaged a two-by-two block of the output;
// the offsets were a quarter of an *input* texel, so all four landed within
// half a pixel of each other and cost four fetches to learn what one would have
// said. Found while closing `PERF-2` by reading the pass that was being made
// cheaper; the sixteen regular taps that replaced them were themselves
// replaced by these four while closing `PERF-5`.
//
// The log is what makes the average a geometric mean, which is what a meter
// wants: one white pixel in a dark room should not open the eye all the way.
//
// **And each tap is snapped to a texel of the source**, which is the half of
// this that took the longest to find. A jitter that lands anywhere reads
// through the linear sampler, so a tap near a stripe's edge comes back as a
// blend of both sides - and log of a blend is above the blend of logs, which is
// a bias upwards on every piece of fine content in the picture and not a
// resonance with any particular one. Snapping keeps the whole of what the
// jitter is for (which texel of two hundred is read varies from output texel to
// output texel) and throws away the part that was never wanted. Measured on
// painted stripes: 0.83 stops out at worst before, 0.18 after.
@fragment
fn fragment_luminance(input: ScreenOutput) -> @location(0) vec4<f32> {
    let step = footprint(input.uv);
    let side = f32(METER_TAPS);
    let size = vec2<f32>(textureDimensions(source));
    var total = 0.0;

    for (var y = 0; y < METER_TAPS; y += 1) {
        for (var x = 0; x < METER_TAPS; x += 1) {
            let cell = vec2<i32>(x, y);
            let place = (vec2<f32>(cell) + scatter(input.uv, cell)) / side - 0.5;
            let at = (floor((input.uv + place * step) * size) + 0.5) / size;
            let bright = luminance(textureSample(source, source_sampler, at).rgb);

            total += log2(max(bright, DARKEST));
        }
    }

    return vec4<f32>(total / f32(METER_TAPS * METER_TAPS), 0.0, 0.0, 1.0);
}

// The same picture at an eighth of the width and an eighth of the height.
//
// **Eight rather than two, because a pass costs more than what is inside it.**
// Measured while closing `PERF-2`: the ladder was nine passes of one tap each
// and cost about fifty-three microseconds, of which five to seven belonged to
// every pass whatever it did. Halving needs eight rungs to get from a
// sixty-four square to one and reducing by eight needs two, so the same
// arithmetic arrives in three passes instead of nine. Godot reduces by eight
// for the same reason (`luminance.cpp:91-93`); Fyrox halves and pays thirteen
// taps a pass, which is the same trade made the other way round.
@fragment
fn fragment_reduce(input: ScreenOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(reduced(input.uv), 0.0, 0.0, 1.0);
}

// Where the eye has got to, one texel - and the ladder's last reduction, in the
// same pass.
//
// **It reduces as well as blends, which saves a pass.** The bottom rung is
// eight texels across and this target is one, so the step between them is
// exactly the reduction every other rung does; doing it here rather than in a
// pass of its own is one fewer barrier for one changed line.
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
    let measured = reduced(input.uv);
    let before = textureLoad(eye, vec2<i32>(0, 0), 0).r;

    return vec4<f32>(mix(before, measured, tuning.meter.w), 0.0, 0.0, 1.0);
}

// What is bright enough to bloom, with a soft edge under the threshold.
//
// A hard cut is what makes bloom flicker: a pixel wandering either side of the
// line appears and disappears whole, and a highlight moving across a wall
// crawls. The knee is a quadratic that eases the contribution in over a band
// below the threshold, which is what Godot and Unity both do and costs four
// lines.
@fragment
fn fragment_threshold(input: ScreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source, source_sampler, input.uv).rgb;
    let bright = max(color.r, max(color.g, color.b));
    let threshold = tuning.bloom.y;
    let knee = max(tuning.bloom.z, 0.0001);

    let soft = clamp(bright - threshold + knee, 0.0, 2.0 * knee);
    let eased = soft * soft / (4.0 * knee);
    let taken = max(eased, bright - threshold) / max(bright, 0.0001);

    return vec4<f32>(color * taken, 1.0);
}

// The same picture at half the width and half the height, in thirteen taps.
//
// A plain four-tap box at half size aliases badly on a highlight one pixel
// wide, and a single bright pixel that appears and vanishes between frames is
// exactly what a bloom chain must not do. The thirteen are four overlapping
// two-by-two groups and a fifth in the middle carrying half the weight, which
// is the filter Jimenez wrote for Call of Duty and everybody ships.
@fragment
fn fragment_down(input: ScreenOutput) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source, 0));
    let uv = input.uv;

    let a = textureSample(source, source_sampler, uv + vec2<f32>(-2.0, -2.0) * t).rgb;
    let b = textureSample(source, source_sampler, uv + vec2<f32>(0.0, -2.0) * t).rgb;
    let c = textureSample(source, source_sampler, uv + vec2<f32>(2.0, -2.0) * t).rgb;
    let d = textureSample(source, source_sampler, uv + vec2<f32>(-2.0, 0.0) * t).rgb;
    let e = textureSample(source, source_sampler, uv).rgb;
    let f = textureSample(source, source_sampler, uv + vec2<f32>(2.0, 0.0) * t).rgb;
    let g = textureSample(source, source_sampler, uv + vec2<f32>(-2.0, 2.0) * t).rgb;
    let h = textureSample(source, source_sampler, uv + vec2<f32>(0.0, 2.0) * t).rgb;
    let i = textureSample(source, source_sampler, uv + vec2<f32>(2.0, 2.0) * t).rgb;

    let j = textureSample(source, source_sampler, uv + vec2<f32>(-1.0, -1.0) * t).rgb;
    let k = textureSample(source, source_sampler, uv + vec2<f32>(1.0, -1.0) * t).rgb;
    let l = textureSample(source, source_sampler, uv + vec2<f32>(-1.0, 1.0) * t).rgb;
    let m = textureSample(source, source_sampler, uv + vec2<f32>(1.0, 1.0) * t).rgb;

    var total = (j + k + l + m) * 0.5;
    total += (a + b + d + e) * 0.125;
    total += (b + c + e + f) * 0.125;
    total += (d + e + g + h) * 0.125;
    total += (e + f + h + i) * 0.125;

    return vec4<f32>(total * 0.25, 1.0);
}

// One rung of the chain spread back over the wider one under it.
//
// A three-by-three tent, and the pass it runs in blends by adding, so a rung
// ends up holding its own light plus everything gathered above it. Widening
// happens here rather than at the downsample because this is the pass that
// runs once per rung on the way back down - which is what turns eight halvings
// into one wide glow instead of eight rings.
@fragment
fn fragment_up(input: ScreenOutput) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source, 0));
    let uv = input.uv;

    var total = textureSample(source, source_sampler, uv).rgb * 4.0;

    total += textureSample(source, source_sampler, uv + vec2<f32>(0.0, -1.0) * t).rgb * 2.0;
    total += textureSample(source, source_sampler, uv + vec2<f32>(-1.0, 0.0) * t).rgb * 2.0;
    total += textureSample(source, source_sampler, uv + vec2<f32>(1.0, 0.0) * t).rgb * 2.0;
    total += textureSample(source, source_sampler, uv + vec2<f32>(0.0, 1.0) * t).rgb * 2.0;

    total += textureSample(source, source_sampler, uv + vec2<f32>(-1.0, -1.0) * t).rgb;
    total += textureSample(source, source_sampler, uv + vec2<f32>(1.0, -1.0) * t).rgb;
    total += textureSample(source, source_sampler, uv + vec2<f32>(-1.0, 1.0) * t).rgb;
    total += textureSample(source, source_sampler, uv + vec2<f32>(1.0, 1.0) * t).rgb;

    return vec4<f32>(total / 16.0, 1.0);
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
    var color = textureSample(source, source_sampler, input.uv).rgb;

    // added before the exposure and the curve, not after: what glows is part
    // of the picture, so it is metered with the rest of it and rolls off the
    // same shoulder. Adding it afterwards would put light on the screen that
    // no exposure could stop down.
    if (tuning.bloom.x > 0.0) {
        color += textureSample(bloom, bloom_sampler, input.uv).rgb * tuning.bloom.x;
    }

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

// The depth the scene wrote, one sample a pixel whatever it drew with: at four
// samples it has been resolved to each pixel's nearest by now (`depth.wgsl`).
// Bound by the view below and by nothing else, at the third binding of its
// group because the first two are the picture and its sampler.
@group(1) @binding(2) var depth: texture_depth_2d;

// How far along the view a stored depth is.
//
// The projection stores `b / d - a` for a point `d` along the view, where `a`
// and `b` are its `z_axis.z` and its `w_axis.z`. This is that line solved for
// `d`, with the two numbers read off the very matrix the frame was drawn with.
fn distance_of(stored: f32) -> f32 {
    return tuning.depth.z / (stored + tuning.depth.y);
}

// A level from nought to one, as the linear value an sRGB target turns back
// into that level on the way out: the curve's own inverse, so that the byte
// that lands is the level times 255.
fn undone(level: f32) -> f32 {
    if (level <= 0.04045) {
        return level / 12.92;
    }

    return pow((level + 0.055) / 1.055, 2.4);
}

// The depth instead of the picture: black at the eye, white `tuning.depth.x`
// along the view from it, straight between, with no exposure and no curve.
//
// **A byte is a distance.** On an sRGB target the curve is undone first, so a
// pixel a quarter of the way to white is 64 in the file, and a color picker
// reads a distance off the picture with one multiplication.
@fragment
fn fragment_depth(input: ScreenOutput) -> @location(0) vec4<f32> {
    let stored = textureLoad(depth, vec2<i32>(input.clip_position.xy), 0);
    let level = clamp(distance_of(stored) / tuning.depth.x, 0.0, 1.0);
    let shown = select(level, undone(level), tuning.depth.w > 0.5);

    return vec4<f32>(shown, shown, shown, 1.0);
}

// What the pass before the scene wrote: xyz a normal in the world, w how rough
// the surface is, and nought in all four wherever nothing was drawn. Bound by
// the view below and by nothing else, at a binding of its own so that it and
// the depth above are never two names for one slot.
@group(1) @binding(3) var surfaces: texture_2d<f32>;

// That buffer instead of the picture: at one the normal as a color, each axis
// from minus one to one laid over nought to one, and at two the roughness as a
// grey. Black wherever nothing was drawn, which no normal comes out as - all
// three of its axes would have to be minus one.
//
// **A byte is a number**, the depth view's rule: on an sRGB target the curve is
// undone first, so an axis of one is 255 in the file and a roughness of a half
// is half of it.
@fragment
fn fragment_surfaces(input: ScreenOutput) -> @location(0) vec4<f32> {
    let held = textureLoad(surfaces, vec2<i32>(input.clip_position.xy), 0);

    if (held.w <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    var level = clamp(held.xyz * 0.5 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));

    if (tuning.depth.x > 1.5) {
        level = vec3<f32>(clamp(held.w, 0.0, 1.0));
    }

    if (tuning.depth.w > 0.5) {
        level = vec3<f32>(undone(level.x), undone(level.y), undone(level.z));
    }

    return vec4<f32>(level, 1.0);
}

// How much of the sky each pixel sees, worked out from what the pass before the
// scene wrote: half the picture on each axis, r the share and g how far along
// the view it was worked out. At a binding of its own for the surfaces' reason.
@group(1) @binding(4) var occlusion: texture_2d<f32>;

// That share instead of the picture, as a grey, each pixel showing the texel it
// is twice the place of - so a pixel at an even place on both axes shows the
// share worked out for exactly it. White where nothing hides the sky and where
// nothing was drawn.
//
// **A byte is a share**, the depth view's rule: on an sRGB target the curve is
// undone first, so half the sky is 128 in the file.
@fragment
fn fragment_occlusion(input: ScreenOutput) -> @location(0) vec4<f32> {
    let held = textureLoad(occlusion, vec2<i32>(input.clip_position.xy) / 2, 0);
    var level = clamp(held.r, 0.0, 1.0);

    if (tuning.depth.w > 0.5) {
        level = undone(level);
    }

    return vec4<f32>(level, level, level, 1.0);
}
