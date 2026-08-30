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
    // Location 11, matching the scene's: where this instance's joint matrices
    // start and how many there are.
    @location(11) skin: vec4<u32>,
};

// The same buffer the scene reads, at group one rather than three: this pass
// has one group of its own and the numbering is per pipeline.
@group(1) @binding(0) var<storage, read> joints: array<mat4x4<f32>>;

struct SkinInput {
    @location(12) bones: vec4<u32>,
    @location(13) weights: vec4<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    return cascade.light_view_projection * (model_of(instance) * vec4<f32>(vertex.position, 1.0));
}

// The same for geometry bones move.
//
// Not an optimization to skip: a character shadowed from its bind pose stands
// in one attitude and is shadowed in another, which is more obviously wrong on
// screen than a missing shadow would be.
@vertex
fn vertex_skinned(
    vertex: VertexInput,
    instance: InstanceInput,
    skin: SkinInput,
) -> @builtin(position) vec4<f32> {
    let posed = skinning(skin, instance.skin.x, instance.skin.y);
    let world = model_of(instance) * (posed * vec4<f32>(vertex.position, 1.0));

    return cascade.light_view_projection * world;
}

// An instance's four columns, put back together.
fn model_of(instance: InstanceInput) -> mat4x4<f32> {
    return mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );
}

// The weighted sum of the four bones that move a vertex. @ref shader.wgsl,
// where the same arithmetic is explained at length; the two are apart because
// a shader module cannot include another one.
fn skinning(skin: SkinInput, at: u32, count: u32) -> mat4x4<f32> {
    if count == 0u {
        return mat4x4<f32>(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0, 0.0, 0.0, 1.0),
        );
    }

    let last = count - 1u;

    return joints[at + min(skin.bones.x, last)] * skin.weights.x
        + joints[at + min(skin.bones.y, last)] * skin.weights.y
        + joints[at + min(skin.bones.z, last)] * skin.weights.z
        + joints[at + min(skin.bones.w, last)] * skin.weights.w;
}
