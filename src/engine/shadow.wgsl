// The depth pass, once per cascade: geometry in, distance from the light out.
//
// Two of the four entry points have no fragment stage at all, which is the
// cheapest possible pass over the same instance buffer the scene draws from: it
// reads the position and the model matrix and ignores every other attribute the
// pipeline supplies. The other two exist because the one thing a fragment here
// can usefully do is *not* happen - a surface whose picture has holes in it has
// to be sampled before it is allowed to write depth, or a fence casts the
// shadow of the sheet it was cut out of.
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
    // Read by the masked entry points and by nothing else. The plain ones
    // declare it and ignore it, which costs nothing: the attribute is the
    // pipeline's and is supplied either way.
    @location(2) uv: vec2<f32>,
};

struct InstanceInput {
    // Locations 4 to 7, matching the scene's: a shader may read fewer
    // attributes than the pipeline declares, and the numbering is the
    // pipeline's rather than this file's.
    @location(4) model_0: vec4<f32>,
    @location(5) model_1: vec4<f32>,
    @location(6) model_2: vec4<f32>,
    @location(7) model_3: vec4<f32>,
    // Location 9, matching the scene's: xy is metallic and roughness, which
    // nothing here reads, and zw is how often the picture repeats, which the
    // masked entry points need or a tiled fence would be sampled once across
    // its whole length.
    @location(9) surface: vec4<f32>,
    // Location 11, matching the scene's: where this instance's joint matrices
    // start and how many there are.
    @location(11) skin: vec4<u32>,
};

// The scene's own material group, at group two rather than one, and declared on
// every pipeline here so that one bind group serves all four. The plain ones do
// not read it; the normal map at binding two is not read by any of them, and a
// shader is allowed to use less of a group than the layout declares.
@group(2) @binding(0) var albedo: texture_2d<f32>;
@group(2) @binding(1) var surface_sampler: sampler;

// The same half `shader.wgsl` compares against, and it has to be the same or a
// fence would cast a shadow with different holes in it than the fence has. A
// module cannot include another one, so there is a test that these two lines
// agree rather than a comment asking somebody to remember.
const MASK_CUTOFF: f32 = 0.5;

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

struct MaskedOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// The same as `vertex_main`, carrying the texture coordinate along.
//
// A separate entry point rather than an output the plain one also writes,
// because an interpolant costs every caster in the world whether the fragment
// stage reads it or not, and almost nothing in a world is masked.
@vertex
fn vertex_masked(vertex: VertexInput, instance: InstanceInput) -> MaskedOutput {
    var output: MaskedOutput;
    output.clip_position =
        cascade.light_view_projection * (model_of(instance) * vec4<f32>(vertex.position, 1.0));
    output.uv = vertex.uv * instance.surface.zw;

    return output;
}

// And the same for geometry bones move.
@vertex
fn vertex_masked_skinned(
    vertex: VertexInput,
    instance: InstanceInput,
    skin: SkinInput,
) -> MaskedOutput {
    let posed = skinning(skin, instance.skin.x, instance.skin.y);

    var output: MaskedOutput;
    output.clip_position =
        cascade.light_view_projection * (model_of(instance) * (posed * vec4<f32>(vertex.position, 1.0)));
    output.uv = vertex.uv * instance.surface.zw;

    return output;
}

// The whole of the fragment stage: a texel that is a hole writes no depth.
//
// It returns nothing and the pipeline has no color target, so being discarded
// is the only thing that can happen here and depth is the only output.
//
// The base level rather than whichever mip a derivative picks, and that is a
// decision rather than a shortcut. The derivative here is against the *light's*
// grid at the map's own resolution, which has nothing to do with how large the
// surface is on somebody's screen; and averaging a cutout down is what turns a
// distant fence into a wall, because the mean of a hole and a solid texel is
// over the cutoff. What it costs is aliasing at distance, which is the honest
// half of the same trade.
@fragment
fn fragment_masked(input: MaskedOutput) {
    if (textureSampleLevel(albedo, surface_sampler, input.uv, 0.0).a < MASK_CUTOFF) {
        discard;
    }
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
