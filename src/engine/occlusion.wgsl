// How much of the light that reaches a surface by other routes each pixel can
// still see, worked out from what the pass before the scene wrote.
//
// Two passes, both at half the picture on each axis. The first follows the
// horizon around every pixel along a few slices of the hemisphere and adds up,
// in closed form, what share of the light a cosine-weighted hemisphere sends
// in over the part of each slice nothing within reach hides. The second
// averages that over the three by three texels around each one that lie on the
// same surface, which is what turns a rotation that steps across a three by
// three tile into none at all. @ref `colby_engine::occlusion`.
//
// **Both read the depth and the normals at the picture's own size.** A texel
// here stands for the pixel at twice its place, and every tap it takes is a
// pixel of the full buffers - nothing is read from a smaller copy of the depth,
// so nothing has to decide which of four depths a smaller copy keeps.

struct Tuning {
    // World space into view space, a row an axis. Only the rotation is read:
    // the normals are directions.
    view_x: vec4<f32>,
    view_y: vec4<f32>,
    view_z: vec4<f32>,
    // x and y are how far the projection scales a view-space x and y at a
    // distance of one; z and w are its `z_axis.z` and `w_axis.z`, the two
    // numbers a stored depth is turned back into a distance with.
    lens: vec4<f32>,
    // x and y are the size of the picture in pixels, z how far away something
    // may be and still hide the sky, w how much of what is hidden is taken away.
    size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

// What the pass before the scene wrote, one sample a pixel.
@group(1) @binding(0) var depth: texture_depth_2d;
@group(1) @binding(1) var surfaces: texture_2d<f32>;

// What the first pass here wrote, read by the second.
@group(2) @binding(0) var raw: texture_2d<f32>;

// How many slices of the hemisphere a texel follows.
//
// Three, and the tile below turns them by one of nine rotations a texel, so a
// surface the average below covers whole has been read along twenty-seven.
const SLICES: u32 = 3u;

// How many taps each side of a slice takes on its way out to the reach.
//
// Spread evenly rather than bunched towards the middle, and each in the middle
// of its step. Bunched is what the field ships, for thin cracks; spread is what
// finds the horizon of something that rises all the way out to the reach, which
// is where an evenly lit surface's own share of the sky is decided - measured
// against the closed form of a crease, evenly spread is half the error at the
// same count. Moving the taps along their steps from texel to texel as well as
// turning the slices was measured too, and read the crease a little worse.
const STEPS: u32 = 8u;

// How wide the tile the rotations step across is, in texels.
const TILE: u32 = 3u;

// How far above a pixel's own plane a tap has to rise, as a share of how far
// away it is, before it hides anything.
//
// A tap is taken at the nearest pixel to where a slice runs, so on a flat
// surface it lands a little off the slice - and a point of the same plane a
// little off the slice reads as a horizon a little above the plane. Taken as
// the most any tap says, which is what a horizon is, that alone darkened an
// open floor by two parts in a hundred. A fiftieth of the distance is about a
// degree: above the fifth of a degree every surface leans by from the flat
// texel of its normal map, and far below anything that hides a share of the
// sky worth a byte.
const RISE: f32 = 0.02;

// The widest the reach may be drawn on the screen, as a share of the picture's
// height.
//
// A surface a hand's width from the eye would otherwise spread its taps over
// the whole picture, each of them a long way from the last.
const WIDEST: f32 = 0.5;

// How far a texel may lie off the plane of the one being averaged and still be
// part of the same surface, as a share of its distance from the eye.
const PLANE: f32 = 0.02;

const PI: f32 = 3.14159265;
const HALF_PI: f32 = 1.57079633;

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
    return tuning.lens.w / (stored + tuning.lens.z);
}

// Where the middle of a pixel of the full picture is in view space, at a
// distance along the view.
fn placed(texel: vec2<i32>, distance: f32) -> vec3<f32> {
    let share = (vec2<f32>(texel) + 0.5) / tuning.size.xy;
    let across = share.x * 2.0 - 1.0;
    let up = 1.0 - share.y * 2.0;

    return vec3<f32>(across * distance / tuning.lens.x, up * distance / tuning.lens.y, -distance);
}

// A direction in the world, turned into view space.
fn into_view(way: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(tuning.view_x.xyz, way),
        dot(tuning.view_y.xyz, way),
        dot(tuning.view_z.xyz, way),
    );
}

// The cosine between the eye and one tap, or `low` for a tap that hides
// nothing: one off the picture, one where nothing was drawn, one out of reach,
// and one that does not rise above the pixel's own plane.
//
// Handing back `low` rather than a flag is what keeps a slice nothing hides
// exactly the number it started as: the most of `low` and `low` is `low`.
fn horizon(
    texel: vec2<i32>,
    here: vec3<f32>,
    normal: vec3<f32>,
    towards_eye: vec3<f32>,
    low: f32,
) -> f32 {
    let size = vec2<i32>(tuning.size.xy);

    if (any(texel < vec2<i32>(0)) || any(texel >= size)) {
        return low;
    }

    let stored = textureLoad(depth, texel, 0);

    if (stored >= 1.0) {
        return low;
    }

    let away = placed(texel, distance_of(stored)) - here;
    let reach = length(away);

    if (reach > tuning.size.z || reach < 1.0e-5) {
        return low;
    }

    if (dot(away, normal) <= RISE * reach) {
        return low;
    }

    return dot(away, towards_eye) / reach;
}

// The cosine-weighted share of a slice between the eye and a horizon at an
// angle `h`, for a normal at an angle `n` in that slice - the closed form of the
// integral over one side, with `h` negative on the side it is negative on.
fn arc(h: f32, n: f32, cos_n: f32, sin_n: f32) -> f32 {
    return (cos_n + 2.0 * h * sin_n - cos(2.0 * h - n)) * 0.25;
}

// Where a texel is in the tile, as a rotation from nought to eight.
fn cell(texel: vec2<i32>) -> u32 {
    // a tile whose neighbors never follow each other, so a texel that only
    // gets part of its window averaged is not left with a run of rotations
    var order = array<u32, 9>(0u, 5u, 7u, 6u, 1u, 3u, 4u, 8u, 2u);
    let at = vec2<u32>(texel) % vec2<u32>(TILE);

    return order[at.y * TILE + at.x];
}

// How much of the hemisphere over one texel's pixel nothing within reach hides:
// r the share, g how far along the view the pixel is, or one and nought where
// nothing was drawn.
@fragment
fn fragment_estimate(input: ScreenOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let center = texel * 2;
    let stored = textureLoad(depth, center, 0);
    let held = textureLoad(surfaces, center, 0);

    if (stored >= 1.0 || held.w <= 0.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }

    let along_view = distance_of(stored);
    let here = placed(center, along_view);
    let normal = normalize(into_view(held.xyz));
    let towards_eye = normalize(-here);

    // a pixel of the full picture per unit at a distance of one along the view
    let focal = tuning.lens.y * tuning.size.y * 0.5;

    let rotation = cell(texel);
    let turn = (f32(rotation) + 0.5) / 9.0;

    var hidden = 0.0;
    var whole = 0.0;

    for (var slice = 0u; slice < SLICES; slice++) {
        let angle = (f32(slice) + turn) * PI / f32(SLICES);
        // a line through the pixel on the screen, and the plane through the eye
        // it lies in: a screen's y runs down where a view's runs up
        let way = vec2<f32>(cos(angle), sin(angle));
        let flat = vec3<f32>(way.x, -way.y, 0.0);
        let along = normalize(flat - towards_eye * dot(flat, towards_eye));

        // how far the reach is on the screen along this slice, in pixels: the
        // projection's own derivative along it, which off the middle of a wide
        // picture is well over what it is along the middle - reckoned along the
        // middle, the taps stopped a quarter short of the reach there
        let stretch = vec2<f32>(
            along.x * -here.z + here.x * along.z,
            along.y * -here.z + here.y * along.z,
        );
        let widest = min(
            tuning.size.z * focal * length(stretch) / (here.z * here.z),
            tuning.size.y * WIDEST,
        );

        let axis = normalize(cross(along, towards_eye));
        let projected = normal - axis * dot(normal, axis);
        let weight = length(projected);

        if (weight < 1.0e-6) {
            continue;
        }

        let cos_n = clamp(dot(projected, towards_eye) / weight, 0.0, 1.0);
        let n = sign(dot(projected, along)) * acos(cos_n);
        let sin_n = sin(n);
        let low_ahead = cos(n + HALF_PI);
        let low_behind = cos(n - HALF_PI);

        var high_ahead = low_ahead;
        var high_behind = low_behind;

        for (var tap = 0u; tap < STEPS; tap++) {
            let reach = max((f32(tap) + 0.5) / f32(STEPS) * widest, 1.0);
            let offset = vec2<i32>(round(way * reach));

            high_ahead = max(high_ahead, horizon(center + offset, here, normal, towards_eye, low_ahead));
            high_behind = max(high_behind, horizon(center - offset, here, normal, towards_eye, low_behind));
        }

        let ahead = acos(clamp(high_ahead, -1.0, 1.0));
        let behind = -acos(clamp(high_behind, -1.0, 1.0));
        let open_ahead = acos(clamp(low_ahead, -1.0, 1.0));
        let open_behind = -acos(clamp(low_behind, -1.0, 1.0));
        let open_ahead_arc = arc(open_ahead, n, cos_n, sin_n);
        let open_behind_arc = arc(open_behind, n, cos_n, sin_n);

        // what each side hides, and nought to the last bit on a side no tap
        // rose on: the same number worked out twice need not come out the same
        // twice once a compiler has had its way with the two expressions, and
        // an open surface came back a bit short of one that way
        let hid_ahead = select(0.0, open_ahead_arc - arc(ahead, n, cos_n, sin_n), high_ahead > low_ahead);
        let hid_behind = select(0.0, open_behind_arc - arc(behind, n, cos_n, sin_n), high_behind > low_behind);

        hidden += weight * (hid_ahead + hid_behind);
        whole += weight * (open_ahead_arc + open_behind_arc);
    }

    // divided by what the same slices add up to with nothing in the way, rather
    // than by one: three slices of an open hemisphere do not add up to exactly
    // one off the middle of the picture
    let share = select(1.0, clamp(1.0 - hidden / whole, 0.0, 1.0), whole > 0.0);

    return vec4<f32>(share, along_view, 0.0, 0.0);
}

// The estimate averaged over the three by three texels around each one that
// lie on its surface, with the strength applied: r what a surface there is
// lit by, as a share, and g the distance carried through.
@fragment
fn fragment_blur(input: ScreenOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let here = textureLoad(raw, texel, 0);

    if (here.g <= 0.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }

    let held = textureLoad(surfaces, texel * 2, 0);
    let normal = normalize(into_view(held.xyz));
    let middle = placed(texel * 2, here.g);
    let last = vec2<i32>(textureDimensions(raw)) - 1;

    // what is hidden rather than what is seen, for the estimate's reason: an
    // average of nine ones came back a bit short of one on this card, where an
    // average of nine noughts cannot come back anything but nought
    var short = 0.0;
    var count = 0.0;

    for (var down = -1; down <= 1; down++) {
        for (var across = -1; across <= 1; across++) {
            let at = texel + vec2<i32>(across, down);

            if (any(at < vec2<i32>(0)) || any(at > last)) {
                continue;
            }

            let other = textureLoad(raw, at, 0);
            let off = abs(dot(placed(at * 2, other.g) - middle, normal));

            if (other.g <= 0.0 || off > PLANE * here.g) {
                continue;
            }

            short += 1.0 - other.r;
            count += 1.0;
        }
    }

    // one strength multiplies what is hidden, so a strength of one hands the
    // average back to the last bit and a surface nothing hides is one at any
    let lit = 1.0 - short / max(count, 1.0) * tuning.size.w;

    return vec4<f32>(lit, here.g, 0.0, 0.0);
}
