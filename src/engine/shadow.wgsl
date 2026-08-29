// The depth pass, once per cascade: geometry in, distance from the light out.
//
// There is no fragment stage at all. The pass writes nothing but depth, so the
// only thing a fragment could do is be discarded, and nothing here is
// transparent or alpha tested. That also makes this the cheapest possible pass
// over the same instance buffer the scene draws from - it reads the position
// and the model matrix and ignores every other attribute the pipeline supplies.
//
// One matrix per cascade, in its own bind group over its own slice of one
// buffer. The alternative is a dynamic offset, which is the same buffer with a
// validation rule attached.

struct Cascade {
    light_view_projection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> cascade: Cascade;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct InstanceInput {
    // Locations 4 to 7, matching the scene's: a shader may read fewer
    // attributes than the pipeline declares, and the numbering is the
    // pipeline's rather than this file's.
    @location(4) model_0: vec4<f32>,
    @location(5) model_1: vec4<f32>,
    @location(6) model_2: vec4<f32>,
    @location(7) model_3: vec4<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );

    return cascade.light_view_projection * (model * vec4<f32>(vertex.position, 1.0));
}
