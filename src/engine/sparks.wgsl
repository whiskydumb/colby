// The particle renderer's whole shader: a camera-facing square, in a color,
// with a picture on it.
//
// It shares group 0 with the scene, so the same uniform buffer and the same
// bind group serve both and the camera cannot disagree between them. Only
// `view_projection` is read, exactly as `lines.wgsl` reads only that, and for
// the same reason: a uniform binding may be larger than the struct reading it,
// and repeating fifteen fields this file never looks at would be fifteen more
// things to keep in step. What has to hold is that `view_projection` stays
// first, and `scene::strides` asserts that.
//
// There is no geometry buffer at all. The four corners come from
// `vertex_index` and the particle comes from `instance_index`, which is what
// makes a cloud of a thousand one draw call and thirty-two bytes each.

struct Globals {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var picture: texture_2d<f32>;
@group(1) @binding(1) var picture_sampler: sampler;

struct Instance {
    // Where the middle of the square is, in world space.
    @location(0) position: vec3<f32>,
    // How wide it is, in world units.
    @location(1) size: f32,
    // Linear RGB and how opaque, both already faded for the particle's age.
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vertex_main(instance: Instance, @builtin(vertex_index) corner: u32) -> VertexOutput {
    // A triangle strip of four: bottom left, bottom right, top left, top
    // right. Two triangles, no index buffer, no vertex buffer.
    let offset = vec2<f32>(
        select(-1.0, 1.0, (corner & 1u) == 1u),
        select(-1.0, 1.0, (corner & 2u) == 2u),
    );

    // The camera's own right and up, in world space, read out of the matrix
    // that is already bound rather than passed again. For view_projection =
    // projection * view with a projection whose upper left is diagonal - which
    // every perspective and every orthographic one is - row 0 is the camera's
    // right times a positive scale and row 1 is its up times another. A
    // column-major mat4x4 spells row 0 as (m[0].x, m[1].x, m[2].x).
    let matrix = globals.view_projection;
    let right = normalize(vec3<f32>(matrix[0].x, matrix[1].x, matrix[2].x));
    let up = normalize(vec3<f32>(matrix[0].y, matrix[1].y, matrix[2].y));

    // Half the width each way, so `size` is the width of the square rather
    // than half of it - which is what somebody typing a number into an
    // inspector means by it.
    let world = instance.position
        + (right * offset.x + up * offset.y) * instance.size * 0.5;

    var out: VertexOutput;

    out.clip_position = matrix * vec4<f32>(world, 1.0);
    out.uv = offset * 0.5 + 0.5;
    out.color = instance.color;

    return out;
}

@fragment
fn fragment_main(fragment: VertexOutput) -> @location(0) vec4<f32> {
    // Unlit on purpose, and the field agrees: Fyrox's particle shader defaults
    // `useLighting` to false and s&box's sprite renderer defaults `Lighting`
    // to false. A particle is a light source about as often as it is a
    // surface, and shading one costs the whole lamp loop per pixel of a thing
    // that is mostly transparent.
    //
    // Every channel of the picture is read, unlike Fyrox's which takes the red
    // one alone: a colored flame that could not be a colored flame would be a
    // strange limit to build in.
    let sampled = textureSample(picture, picture_sampler, fragment.uv);

    return sampled * fragment.color;
}
