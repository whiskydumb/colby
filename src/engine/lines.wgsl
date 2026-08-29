// The debug renderer's whole shader: a segment, in world space, in a color.
//
// It shares group 0 with the scene, so the same uniform buffer and the same
// bind group serve both and the camera cannot disagree between them. Only
// `view_projection` is read, and a uniform binding may be larger than the
// struct reading it, so the rest of the scene's Globals is deliberately not
// repeated here - four light matrices this file never looks at would be four
// more things to keep in step. What has to hold is that `view_projection` stays
// first, and `scene::strides` asserts exactly that.

struct Globals {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

struct VertexInput {
    @location(0) position: vec3<f32>,
    // Linear RGB, like every other color the renderer is handed.
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = globals.view_projection * vec4<f32>(vertex.position, 1.0);
    out.color = vertex.color;

    return out;
}

@fragment
fn fragment_main(fragment: VertexOutput) -> @location(0) vec4<f32> {
    // Unlit on purpose. A debug line is a statement about geometry, and shading
    // it would make its color a function of where it happens to be pointing.
    return vec4<f32>(fragment.color, 1.0);
}
