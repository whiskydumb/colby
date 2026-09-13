// What the reflections found, made into something the picture can read: an
// average over the texels around each one that lie on its surface, and then
// the whole picture's size again.
//
// The finding itself is in `shader.wgsl`, because it lights what a reflection
// meets with the scene's own arithmetic. This file is the half that needs none
// of that: two passes over buffers of light that are already multiplied by how
// much of each reflection they stand for, so an average of them is an average
// of light. @ref `colby_engine::reflection`.
//
// **Both read the depth and the normals at the picture's own size**, as the
// occlusion's passes do, and for their reason: nothing has to decide which of
// four depths a smaller copy keeps.

// Laid out the way `colby_engine::reflection::Tuning` writes it, and the way
// `Mirror` in `shader.wgsl` reads the same block.
struct Tuning {
    // World space into view space, a row an axis. Only the rotation is read:
    // the normals are directions.
    view_x: vec4<f32>,
    view_y: vec4<f32>,
    view_z: vec4<f32>,
    // x and y how far the projection scales a view-space x and y at a distance
    // of one; z and w the two numbers a stored depth is turned back into a
    // distance with.
    lens: vec4<f32>,
    // x and y the size of the whole target in pixels; w the strength.
    size: vec4<f32>,
    // The rectangle of the target the picture is drawn into: x, y, width,
    // height, in pixels.
    rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tuning: Tuning;

// What the pass before the scene wrote, one sample a pixel.
@group(1) @binding(0) var depth: texture_depth_2d;
@group(1) @binding(1) var surfaces: texture_2d<f32>;

// What the pass before this one wrote.
@group(2) @binding(0) var source: texture_2d<f32>;

// The roughness at and under which a texel is not averaged with anything.
//
// A mirror's texel reflects exactly what it reflects, and the texel beside it
// reflects the thing beside that: averaging the two would blur a mirror by the
// size of three texels for nothing, since every direction a smooth surface's
// lobe draws is the mirror direction to within a fraction of a degree.
const SHARP: f32 = 0.1;

// The roughness at and past which a texel is averaged with all eight around it
// that lie on its surface, and between the two by a share of them.
const SPREAD: f32 = 0.3;

// How far a texel may lie off the plane of the one it is read for and still be
// part of the same surface, as a share of its distance from the eye: the
// occlusion's number, for the occlusion's reason.
const PLANE: f32 = 0.02;

// How far its normal may turn from the one it is read for, as the cosine
// between the two, and still be part of the same surface.
//
// **The plane alone is not enough where two surfaces meet**: the floor at the
// foot of a pillar lies in the plane of the pillar's face, and a texel of the
// pillar that took it for its own would take what the floor found. A quarter of
// a right angle is well past what a ball turns by from one texel to the next
// and well short of any corner.
const TURN: f32 = 0.9;

// The roughness at and past which nothing was followed and nothing is
// averaged. `MIRROR_CUTOFF` in `shader.wgsl`, and a test says the two agree.
const CUTOFF: f32 = 0.6;

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

// Where the middle of a pixel of the target is in view space, at a distance
// along the view, measured across the rectangle the picture is drawn into.
fn placed(pixel: vec2<i32>, distance: f32) -> vec3<f32> {
    let share = (vec2<f32>(pixel) + 0.5 - tuning.rect.xy) / tuning.rect.zw;
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

// The first texel of the half-sized buffers the rectangle drawn into covers.
fn first_texel() -> vec2<i32> {
    return vec2<i32>(tuning.rect.xy) / 2;
}

// And the last, which is what a neighbor past the rectangle's edge is read as:
// a picture drawn into the middle of a window reads at its edges what the same
// picture drawn alone reads at the target's.
fn last_texel() -> vec2<i32> {
    return (vec2<i32>(tuning.rect.xy + tuning.rect.zw) + 1) / 2 - 1;
}

// Whether the pixel at `other` is part of the surface through `middle`, at a
// distance of `along` from the eye, facing `normal`: on its plane, facing the
// same way, and on the same side of the cutoff - a rough patch on a mirror lies
// on the mirror's plane and faces its way, and followed nothing.
fn on_plane(other: vec2<i32>, middle: vec3<f32>, normal: vec3<f32>, along: f32) -> bool {
    let stored = textureLoad(depth, other, 0);

    if (stored >= 1.0) {
        return false;
    }

    let held = textureLoad(surfaces, other, 0);
    let off = abs(dot(placed(other, distance_of(stored)) - middle, normal));
    let facing = dot(normalize(into_view(held.xyz)), normal);

    return off <= PLANE * along && facing >= TURN && held.w < CUTOFF;
}

// Each texel averaged with the eight around it that lie on its surface, by as
// much of each as its roughness says: none of them at `SHARP` and under, all of
// them at `SPREAD` and over.
//
// **A texel not averaged comes back to the last bit**: nothing is added to it,
// and it is divided by exactly one.
@fragment
fn fragment_average(input: ScreenOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let own = textureLoad(source, texel, 0);
    let pixel = texel * 2;
    let stored = textureLoad(depth, pixel, 0);
    let held = textureLoad(surfaces, pixel, 0);
    let spread = clamp((held.w - SHARP) / (SPREAD - SHARP), 0.0, 1.0);

    if (stored >= 1.0 || held.w <= 0.0 || held.w >= CUTOFF || spread <= 0.0) {
        return own;
    }

    let along = distance_of(stored);
    let middle = placed(pixel, along);
    let normal = normalize(into_view(held.xyz));
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

            if (!on_plane(at * 2, middle, normal, along)) {
                continue;
            }

            total += textureLoad(source, at, 0) * spread;
            weight += spread;
        }
    }

    return total / weight;
}

// The half-sized buffer at the whole picture's size: each pixel blends the four
// texels around it by where it lies between them, and a texel that is not on
// its surface is read as whichever of the ones that are is nearest it in
// distance.
//
// The occlusion's upsample, done here once rather than in the scene's fragment
// stage, and with the plane rather than a slope: this pass has the pixel's own
// normal to hand. **A pixel at an even place on both axes reads its own texel
// and nothing else**, to the last bit.
@fragment
fn fragment_upsample(input: ScreenOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.clip_position.xy);
    let stored = textureLoad(depth, pixel, 0);
    let held = textureLoad(surfaces, pixel, 0);

    // nothing drawn, and nothing followed: a pixel as rough as the cutoff reads
    // nothing whatever the texels around it found
    if (stored >= 1.0 || held.w <= 0.0 || held.w >= CUTOFF) {
        return vec4<f32>(0.0);
    }

    let first = first_texel();
    let last = last_texel();
    let base = pixel / 2;
    let t = vec2<f32>(pixel - base * 2) * 0.5;
    let places = array<vec2<i32>, 4>(
        clamp(base, first, last),
        clamp(base + vec2<i32>(1, 0), first, last),
        clamp(base + vec2<i32>(0, 1), first, last),
        clamp(base + vec2<i32>(1, 1), first, last),
    );

    var texels = array<vec4<f32>, 4>(
        textureLoad(source, places[0], 0),
        textureLoad(source, places[1], 0),
        textureLoad(source, places[2], 0),
        textureLoad(source, places[3], 0),
    );

    // four texels that found nothing blend to nothing whatever the plane says,
    // and on a world with little to reflect that is nearly every pixel
    if (texels[0].a <= 0.0 && texels[1].a <= 0.0 && texels[2].a <= 0.0 && texels[3].a <= 0.0) {
        return vec4<f32>(0.0);
    }

    let along = distance_of(stored);
    let here = placed(pixel, along);
    let normal = normalize(into_view(held.xyz));

    var own = array<bool, 4>();
    var nearest = 0u;
    var closest = 3.4e38;
    var nearest_own = 4u;
    var closest_own = 3.4e38;

    for (var index = 0u; index < 4u; index++) {
        let at = places[index];
        let other = textureLoad(depth, at * 2, 0);
        let gap = abs(distance_of(other) - along);

        own[index] = on_plane(at * 2, here, normal, along);

        if (gap < closest) {
            closest = gap;
            nearest = index;
        }

        if (own[index] && gap < closest_own) {
            closest_own = gap;
            nearest_own = index;
        }
    }

    // a texel of the pixel's own surface stands in for one that is not, and
    // only where none of the four is on it does the nearest in distance: at the
    // foot of a pillar the floor's texel is as far away as the pillar's own, and
    // the nearest in distance would be the floor
    let stand_in = select(nearest, nearest_own, nearest_own < 4u);

    for (var index = 0u; index < 4u; index++) {
        if (!own[index]) {
            texels[index] = texels[stand_in];
        }
    }

    let upper = texels[0] + (texels[1] - texels[0]) * t.x;
    let lower = texels[2] + (texels[3] - texels[2]) * t.x;

    return upper + (lower - upper) * t.y;
}
