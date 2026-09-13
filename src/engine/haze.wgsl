// What the march through the air found, averaged over the texels around each
// one that stand for the same air.
//
// The march itself is in `shader.wgsl`, because it lights the air with the
// scene's own lamps and their maps, and so is the pass that puts the result
// over the picture, which reads the environment. This file is the half that
// needs none of that. @ref `colby_engine::haze`.

// Laid out the way `colby_engine::haze::Tuning` writes it, and the way `Air` in
// `shader.wgsl` reads the same block.
struct Tuning {
    // x the density, y how far the air goes on, z and w the two numbers a
    // stored depth is turned back into a distance with.
    medium: vec4<f32>,
    // x and y the size of the whole target in pixels.
    size: vec4<f32>,
    // The rectangle of the target the picture is drawn into: x, y, width,
    // height, in pixels.
    rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

// The depth the scene wrote, one sample a pixel.
@group(1) @binding(0) var depth: texture_depth_2d;

// What the march wrote.
@group(2) @binding(0) var source: texture_2d<f32>;

// How far a neighbor's distance may be from the texel's own, as a share of the
// texel's, and still be averaged with it: `AIR_PLANE` in `shader.wgsl`, and a
// test says the two agree.
const PLANE: f32 = 0.05;

// How far a neighbor's light may be from the texel's own, as a share of the
// brighter of the two, and still be averaged with it.
//
// **The part of this that is not a depth test, and the one a beam needs.** Where
// the march spread its places over the tile, two texels of the same air differ
// by a few percent; where a shadow crosses the air, by most of the light.
//
// Measured with the engine at seven-twenty, against the same march at five
// hundred and twelve places and nothing averaged. A narrow cone cut by a slat
// had 22 pixels past two levels unaveraged, 3,066 averaged by the depth alone,
// and 315 told by the light as well; a point lamp's glow, where only the tile's
// pattern is there to take out, had 3,219 unaveraged and 39 averaged. **The
// average costs a beam's edges what it buys a glow**, and a quarter is where the
// two meet: a tenth left the cone 39 and the glow 864, a half 1,282 and 16.
const CLOSE: f32 = 0.25;

struct ScreenOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// One triangle covering the target, out of nothing but the vertex index.
@vertex
fn vertex_screen(@builtin(vertex_index) index: u32) -> ScreenOutput {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;

    var output: ScreenOutput;
    output.clip_position = vec4<f32>(x, y, 0.0, 1.0);

    return output;
}

// How far along the view a stored depth is.
fn distance_of(stored: f32) -> f32 {
    return tuning.medium.w / (stored + tuning.medium.z);
}

// How bright a light is, by the weights a screen gives its three channels.
fn brightness(light: vec3<f32>) -> f32 {
    return dot(light, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// The first texel of the half-sized buffer the rectangle drawn into covers.
fn first_texel() -> vec2<i32> {
    return vec2<i32>(tuning.rect.xy) / 2;
}

// And the last, which is what a neighbor past the rectangle's edge is not read
// past: a picture drawn into the middle of a window averages at its edges what
// the same picture drawn alone averages at the target's.
fn last_texel() -> vec2<i32> {
    return (vec2<i32>(tuning.rect.xy + tuning.rect.zw) + 1) / 2 - 1;
}

// Each texel averaged with the eight around it that stand for the same air: in
// front of a surface as far away as its own, and lit about as much.
//
// **A texel with no neighbor like it comes back to the last bit**: nothing is
// added to it, and it is divided by exactly one.
@fragment
fn fragment_average(input: ScreenOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let own = textureLoad(source, texel, 0).rgb;
    let along = distance_of(textureLoad(depth, texel * 2, 0));
    let bright = brightness(own);
    let first = first_texel();
    let last = last_texel();
    var total = own;
    var weight = 1.0;

    for (var down = -1; down <= 1; down++) {
        for (var across = -1; across <= 1; across++) {
            let at = texel + vec2<i32>(across, down);

            if ((across == 0 && down == 0) || any(at < first) || any(at > last)) {
                continue;
            }

            let other = distance_of(textureLoad(depth, at * 2, 0));
            let light = textureLoad(source, at, 0).rgb;
            let theirs = brightness(light);
            let near = abs(other - along) <= PLANE * along;
            let alike = abs(theirs - bright) <= CLOSE * max(theirs, bright);

            if (near && alike) {
                total += light;
                weight += 1.0;
            }
        }
    }

    return vec4<f32>(total / weight, 1.0);
}
