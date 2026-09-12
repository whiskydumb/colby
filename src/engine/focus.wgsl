// The lens out of focus, as three passes over the picture.
//
// **The circle of confusion is a ramp, not a lens equation.** How far a pixel
// is from the plane in focus, over how far off that plane blur is complete,
// times a radius in pixels: three numbers a person can see in the picture,
// where a physical lens would want a sensor size this camera does not have.
//
// **The first two passes run at half the picture on each axis**, which is the
// smallest reduction anything in the field takes for this. The reduction is
// free: one linear tap at a half-size texel's center lands exactly between
// four texels of the full-size picture, which is their average.
//
// **The third pass never reads the picture it writes into.** It hands the
// blend the blurred color and, as alpha, how much of it belongs here - so
// `blurred * a + picture * (1 - a)` is done by the hardware, and a pixel in
// focus is left exactly as it was rather than copied through a blur.

// x is how far away the lens is focused; y is how far off that a surface is
// blurred all the way; z is how wide the blur gets there, as a radius in
// pixels of the picture; w is unused.
// x and y are the two numbers of the projection a stored depth is turned back
// into a distance with, its `z_axis.z` and its `w_axis.z`; z and w are unused.
// x and y are one texel of the half-size buffer, as a share of the picture; z
// and w are unused.
struct Tuning {
    lens: vec4<f32>,
    range: vec4<f32>,
    texel: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

@group(1) @binding(0) var source: texture_2d<f32>;
@group(1) @binding(1) var source_sampler: sampler;

// The depth the scene wrote, one sample a pixel whatever it drew with. Bound
// by the first pass and by the last, and not by the one between them: that one
// reads the circle of confusion out of the alpha the first wrote, which is a
// texel it is sampling anyway.
@group(2) @binding(0) var depth: texture_depth_2d;

// How many pixels of the picture one pixel of the blur stands for, per axis.
//
// Matched by `SCALE` in `focus.rs`, and the two have to agree: this is what
// turns a pixel of the blur back into the pixel of the depth buffer at the
// middle of the block it covers.
const SCALE: i32 = 2;

// The most taps the march takes on each side of a pixel.
//
// Matched by `MAX_BLUR` in `focus.rs`, which clamps the radius a world may ask
// for to twice this: a tap here is a texel of the half-size buffer and a texel
// there is two pixels of the picture. It is a bound on the loop rather than a
// look - the radius a pixel actually uses is its own, and a pixel in focus
// takes one tap.
const TAPS: i32 = 32;

struct ScreenOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One triangle covering the target, out of nothing but the vertex index.
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

// How far along the view a stored depth is.
//
// The projection stores `b / d - a` for a point `d` along the view, where `a`
// and `b` are its `z_axis.z` and its `w_axis.z`. This is that line solved for
// `d`, and it is the same line `post.wgsl` and `shaft.wgsl` read the depth
// with.
fn distance_of(stored: f32) -> f32 {
    return tuning.range.y / (stored + tuning.range.x);
}

// How wide the blur is at a pixel, as a radius in pixels of the picture.
//
// A straight ramp away from the plane in focus, the same on both sides of it
// and clamped at the far end. **What wrote no depth is at the far plane**, so
// the sky takes the widest blur there is - which is right, the sky being as
// far away as anything gets, and invisible, a gradient being what it is.
//
// @param stored - what the depth buffer holds at this pixel
fn confusion(stored: f32) -> f32 {
    let range = max(tuning.lens.y, 1.0e-6);
    let away = abs(distance_of(stored) - tuning.lens.x) / range;

    return tuning.lens.z * clamp(away, 0.0, 1.0);
}

// One direction of a separable gaussian, at a radius this pixel chose.
//
// The radius is two standard deviations, which is the usual relation, and the
// taps run out to it: a gaussian cut at two sigma keeps about nineteen parts
// in twenty, and what is dropped is divided out again by the sum of the
// weights rather than left as a dimming.
//
// **`textureSampleLevel` rather than `textureSample`**, because the loop's
// bound is this pixel's own and a sample that works out its own mip level may
// not be taken where neighboring pixels disagree about whether to take it.
//
// @param uv - where in the source this pixel is
// @param coc - the circle of confusion here, as a radius in pixels
// @param step - one texel of the half-size buffer along the axis being blurred
fn along(uv: vec2<f32>, coc: f32, step: vec2<f32>) -> vec3<f32> {
    // the radius in texels of the half-size buffer, which is half what it is
    // in pixels of the picture
    let radius = coc * 0.5;
    let support = min(i32(ceil(radius)), TAPS);
    let sigma = max(radius * 0.5, 1.0e-6);
    let falloff = -1.0 / (2.0 * sigma * sigma);

    var total = textureSampleLevel(source, source_sampler, uv, 0.0).rgb;
    var weights = 1.0;

    for (var tap = 1; tap <= support; tap += 1) {
        let away = step * f32(tap);
        let weight = exp(falloff * f32(tap) * f32(tap));

        total += (textureSampleLevel(source, source_sampler, uv + away, 0.0).rgb
            + textureSampleLevel(source, source_sampler, uv - away, 0.0).rgb) * weight;
        weights += weight * 2.0;
    }

    return total / weights;
}

// The picture blurred across, at half the width and half the height.
//
// Reduced and blurred in one pass: the source is the picture itself and every
// tap is a half-size texel wide, so the tap at the middle is the average of
// the four pixels this one stands for and the ones either side are the
// averages of theirs.
//
// The depth is read at the middle of that block rather than averaged over it -
// a blur has no detail in it by construction - and the circle of confusion it
// gives is written into the alpha, where the pass after this one reads it
// without binding the depth at all.
@fragment
fn fragment_across(input: ScreenOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(input.clip_position.xy) * SCALE + SCALE / 2;
    let coc = confusion(textureLoad(depth, at, 0));

    return vec4<f32>(along(input.uv, coc, vec2<f32>(tuning.texel.x, 0.0)), coc);
}

// The same, down, at the same size.
//
// The two passes are the two halves of one separable blur, and the radius the
// second uses is the radius the first wrote - not the radius of the texels it
// is reading. A separable blur whose width varies from pixel to pixel is not
// exactly a two-dimensional gaussian, and the field's cheap path is this one.
@fragment
fn fragment_down(input: ScreenOutput) -> @location(0) vec4<f32> {
    let coc = textureSampleLevel(source, source_sampler, input.uv, 0.0).a;

    return vec4<f32>(along(input.uv, coc, vec2<f32>(0.0, tuning.texel.y)), coc);
}

// The blur back over the picture, mixed in by how far out of focus each pixel
// is.
//
// **The alpha is the mix and the blend does it**, which is what lets this pass
// put a blur over a picture it is not allowed to read. A circle of confusion
// under one pixel wide mixes in proportionally and nothing at all at nought,
// so a pixel on the plane in focus comes out of this pass exactly as it went
// into the frame.
//
// The depth is read at full size here rather than at half: the mix is where
// the sharp and the blurred halves of the picture meet, and a mix worked out
// at half the resolution would put a stair on every such edge.
@fragment
fn fragment_over(input: ScreenOutput) -> @location(0) vec4<f32> {
    let coc = confusion(textureLoad(depth, vec2<i32>(input.clip_position.xy), 0));
    let color = textureSampleLevel(source, source_sampler, input.uv, 0.0).rgb;

    return vec4<f32>(color, clamp(coc, 0.0, 1.0));
}
