// The light the air catches around the sun, as three passes over the picture.
//
// **Nothing here knows what a sun looks like.** The light it smears is what
// the picture already holds where nothing was drawn - the sky, in a world with
// one - so the first pass is a mask cut out of the picture by the depth, the
// second drags that mask outwards from where the sun is on the screen, and the
// third adds the result back. A world with no sky smears its clear color, which
// is the same sentence.
//
// **The first two run at a quarter of the picture on each axis.** A smear has
// no detail in it by construction, and a sixteenth of the pixels is a
// sixteenth of the taps.

// x is where the sun is across the picture and y is down it, in the
// coordinates a tap is taken at; z is how far of that anything happens at all,
// as a share of the picture's height; w is the picture's width over its
// height, which is what makes that share a circle rather than an ellipse.
// x is how far along the way to the sun the march reaches; y is how much one
// tap adds; z is how much less the next one adds; w is how much of the whole
// is put back, with the fade by how squarely the sun is faced already in it.
// x is how far along the view a surface has to be before its light counts; y
// and z are the projection's two numbers a stored depth is turned back into a
// distance with; w is unused.
struct Tuning {
    sun: vec4<f32>,
    smear: vec4<f32>,
    range: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

@group(1) @binding(0) var source: texture_2d<f32>;
@group(1) @binding(1) var source_sampler: sampler;

// The depth the scene wrote, one sample a pixel whatever it drew with. Bound
// by the mask alone, which is why it is a group of its own: the two passes
// after it read a picture and nothing else.
@group(2) @binding(0) var depth: texture_depth_2d;

// How many taps the smear takes towards the sun.
//
// Matched by `TAPS` in `shaft.rs`, which needs it to work out what the march
// sums to. The count and the three numbers in `tuning.smear` are a look rather
// than a choice a world makes, so they are fixed rather than being fields on a
// record. Sixty-four is what the one engine in the field that ships this shape
// uses, with the same reach, weight and decay.
const TAPS: i32 = 64;

// How many pixels of the picture one pixel of the mask stands for, per axis.
//
// Matched by `SCALE` in `shaft.rs`, and the two have to agree: this is what
// turns a pixel of the mask back into the pixel of the depth buffer at the
// middle of the block it covers.
const SCALE: i32 = 4;

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
// `d`, and it is the same line `post.wgsl` reads the depth view with.
fn distance_of(stored: f32) -> f32 {
    return tuning.range.z / (stored + tuning.range.y);
}

// How much a point of the picture counts as light on its way to the eye.
//
// Two things take it away. **Distance**: only what is past the middle of the
// view counts, easing in from nothing there to all of it at the far plane, so
// that a wall crossing the line fades rather than appears. Where nothing was
// drawn at all the depth is the clear, which is the far plane exactly, and that
// is the sky - it writes no depth of its own, so it needs no special case here.
// **The edges of the picture**: what is about to enter the frame from the side
// has not been drawn yet, and letting the edge count makes a smear jump the
// moment a wall slides in. Both are the field's own shape, the second in the
// fourth power.
//
// @param uv - where in the picture this is, nought to one on each axis
// @param stored - what the depth buffer holds here
fn reaching(uv: vec2<f32>, stored: f32) -> f32 {
    let middle = max(tuning.range.x, 0.0001);
    let far = clamp((distance_of(stored) - middle) / middle, 0.0, 1.0);

    var edge = 1.0 - uv.x * (1.0 - uv.x) * uv.y * (1.0 - uv.y) * 8.0;
    edge = edge * edge;
    edge = edge * edge;

    return far * (1.0 - edge);
}

// How much of the smear happens this far from the sun.
//
// One over the distance would reach the whole picture and wash it; this is the
// field's answer, a straight fall to nothing over a circle around the sun,
// squared so that the middle of it keeps its strength. The circle is measured
// in pictures high rather than in texture coordinates, which is what
// `tuning.sun.w` is for - without it the circle is as wide as the picture is.
//
// @param uv - where in the picture this is
fn around(uv: vec2<f32>) -> f32 {
    let wide = max(tuning.sun.z, 0.0001);
    let away = length((uv - tuning.sun.xy) * vec2<f32>(tuning.sun.w, 1.0));
    let inside = 1.0 - clamp(away / wide, 0.0, 1.0);

    return inside * inside;
}

// The picture, cut down to what the air is allowed to catch.
//
// One tap of the picture and one of the depth, at a quarter of the width and a
// quarter of the height. The depth is read at the middle of the block this
// pixel stands for rather than averaged over it: a mask that is about to be
// smeared sixty-four ways does not repay sixteen loads a pixel.
@fragment
fn fragment_mask(input: ScreenOutput) -> @location(0) vec4<f32> {
    let at = vec2<i32>(input.clip_position.xy) * SCALE + SCALE / 2;
    let stored = textureLoad(depth, at, 0);
    let color = textureSample(source, source_sampler, input.uv).rgb;
    let taken = reaching(input.uv, stored) * around(input.uv);

    return vec4<f32>(color * taken, 1.0);
}

// The mask dragged outwards from the sun.
//
// A march from this pixel towards the sun, each step adding less than the one
// before it, which is what makes a bright gap in a wall a ray rather than a
// disc. The straight line between two points in these coordinates is the
// straight line between them on the screen, so nothing here is corrected for
// the shape of the picture - only the circle in `around` is, because that one
// is a distance.
@fragment
fn fragment_smear(input: ScreenOutput) -> @location(0) vec4<f32> {
    let step = (input.uv - tuning.sun.xy) * tuning.smear.x / f32(TAPS);
    var at = input.uv;
    var color = textureSample(source, source_sampler, at).rgb;
    var fading = 1.0;

    for (var tap = 0; tap < TAPS; tap += 1) {
        at -= step;
        color += textureSample(source, source_sampler, at).rgb * fading * tuning.smear.y;
        fading *= tuning.smear.z;
    }

    return vec4<f32>(color, 1.0);
}

// The smear, back over the picture.
//
// The pass this runs in adds rather than replaces, and it runs before the eye
// measures anything and before the bloom gathers - so a strong shaft stops the
// eye down and glows at its edges, the way the light it stands for would. The
// upscale from a quarter is the sampler's.
@fragment
fn fragment_apply(input: ScreenOutput) -> @location(0) vec4<f32> {
    let color = textureSample(source, source_sampler, input.uv).rgb;

    return vec4<f32>(color * tuning.smear.w, 1.0);
}
