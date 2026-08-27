// One pipeline, one draw call per (mesh, material) pair, one instance per
// entity. The mesh supplies geometry, the instance supplies a model matrix and
// the material's numbers, group 1 supplies the albedo texture, and the globals
// supply the camera and the light.
//
// Shading is metallic-roughness: Cook-Torrance specular with GGX, Smith
// visibility and Schlick's Fresnel, over a Lambert diffuse. One directional
// light, no shadows, no image-based lighting - the ambient term stands in for
// everything the scene does not simulate, which is why it is a color and not a
// number.

struct Globals {
    view_projection: mat4x4<f32>,
    // xyz is the direction the light travels; w is unused.
    light: vec4<f32>,
    // rgb is how lit a surface facing away from the light still is.
    ambient: vec4<f32>,
    // xyz is where the camera is; w is unused.
    eye: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Origin top left; the importer flips OBJ's bottom-up v on the way in.
    @location(2) uv: vec2<f32>,
};

struct InstanceInput {
    // A model matrix, one column per location. wgsl has no matrix vertex
    // attribute, so it arrives as four vectors and is put back together here.
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    // The material's base color times the entity's own tint.
    @location(7) tint: vec4<f32>,
    // x is metallic, y is roughness, zw is how often the texture repeats.
    @location(8) surface: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) tint: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) surface: vec2<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(
        instance.model_0,
        instance.model_1,
        instance.model_2,
        instance.model_3,
    );

    let world_position = model * vec4<f32>(vertex.position, 1.0);

    var output: VertexOutput;
    output.clip_position = globals.view_projection * world_position;
    // @note: the model matrix rather than its inverse transpose. That is only
    // correct while scale is uniform, which is all anything spawns so far.
    // Non-uniform scale will need the real normal matrix.
    output.normal = (model * vec4<f32>(vertex.normal, 0.0)).xyz;
    output.tint = instance.tint.rgb;
    output.uv = vertex.uv * instance.surface.zw;
    output.world_position = world_position.xyz;
    output.surface = instance.surface.xy;

    return output;
}

// How much of the surface's microfacets point along the half vector.
// Trowbridge-Reitz, which everyone calls GGX.
fn distribution_ggx(normal_dot_half: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = normal_dot_half * normal_dot_half * (a2 - 1.0) + 1.0;

    return a2 / max(3.14159265 * d * d, 0.0001);
}

// How much of them shadow each other, Smith's height-correlated form, already
// divided by the 4 * n.l * n.v the specular term would otherwise need.
fn visibility_smith(normal_dot_view: f32, normal_dot_light: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let view = normal_dot_light * sqrt(normal_dot_view * normal_dot_view * (1.0 - a2) + a2);
    let light = normal_dot_view * sqrt(normal_dot_light * normal_dot_light * (1.0 - a2) + a2);

    return 0.5 / max(view + light, 0.0001);
}

// How reflective the surface is at this angle. Schlick's approximation.
fn fresnel_schlick(view_dot_half: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - view_dot_half, 0.0, 1.0), 5.0);
}

// The same, for a whole hemisphere of incoming light rather than one direction.
// A rough surface cannot reflect a sharp rim, so the term it grows towards is
// held down by the roughness instead of going all the way to white.
fn fresnel_ambient(normal_dot_view: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let ceiling = max(vec3<f32>(1.0 - roughness), f0);

    return f0 + (ceiling - f0) * pow(clamp(1.0 - normal_dot_view, 0.0, 1.0), 5.0);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // the texture is sRGB, so this is already linear by the time it is a float.
    // A material with no image samples the one white texel and multiplies by
    // one, which is why there is no branch here.
    let sampled = textureSample(albedo, albedo_sampler, input.uv);
    let base_color = input.tint * sampled.rgb;

    let metallic = clamp(input.surface.x, 0.0, 1.0);
    // clamped away from zero: a perfect mirror makes the GGX denominator
    // vanish, and the highlight becomes a single blinding pixel.
    let roughness = clamp(input.surface.y, 0.045, 1.0);

    let normal = normalize(input.normal);
    let towards_light = normalize(-globals.light.xyz);
    let towards_eye = normalize(globals.eye.xyz - input.world_position);
    let half_vector = normalize(towards_light + towards_eye);

    let normal_dot_light = max(dot(normal, towards_light), 0.0);
    let normal_dot_view = max(dot(normal, towards_eye), 0.0001);
    let normal_dot_half = max(dot(normal, half_vector), 0.0);
    let view_dot_half = max(dot(towards_eye, half_vector), 0.0);

    // a dielectric reflects four percent head on and is white doing it; a metal
    // reflects its own color and has no diffuse term at all.
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let diffuse_color = base_color * (1.0 - metallic);

    let fresnel = fresnel_schlick(view_dot_half, f0);
    let specular = fresnel
        * distribution_ggx(normal_dot_half, roughness)
        * visibility_smith(normal_dot_view, normal_dot_light, roughness);

    let diffuse = (vec3<f32>(1.0) - fresnel) * diffuse_color / 3.14159265;
    let direct = (diffuse + specular) * normal_dot_light * 3.14159265;

    // everything this renderer does not simulate, in one term, standing in for
    // an environment there is no map of.
    //
    // @note: the specular half of it matters more than it looks. A metal has no
    // diffuse term at all, so without this a gold cube under one light is black
    // everywhere the highlight is not - physically right, and it reads as a bug.
    // A prefiltered environment is the real answer; this is the placeholder
    // every renderer uses before it has one.
    let ambient_specular = fresnel_ambient(normal_dot_view, f0, roughness);
    let indirect = globals.ambient.rgb * (diffuse_color + ambient_specular);

    return vec4<f32>(direct + indirect, 1.0);
}
