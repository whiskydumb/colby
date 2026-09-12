// One draw call per (mesh, material) pair, one instance per entity. The mesh
// supplies geometry, the instance supplies a model matrix and the material's
// numbers, group 1 supplies the albedo texture, and the globals supply the
// camera and the light.
//
// There are two vertex entry points and three fragment ones, and that is not a
// coincidence: bones and alpha are independent axes, and a pipeline's vertex
// buffers, its early depth test and its blending are all fixed when it is
// built, so none of the three can be a branch. Six pipelines come out of the
// six pairs.
//
// Shading is metallic-roughness: Cook-Torrance specular with GGX, Smith
// visibility and Schlick's Fresnel, over a Lambert diffuse. One directional
// light with cascaded shadows, up to MAX_LAMPS local ones with none, and no
// image-based lighting - the ambient term stands in for everything the scene
// does not simulate, which is why it is a color and not a number.
//
// The local lights are a flat array walked by every fragment: no tiles, no
// clusters, no per-object list. What keeps that affordable is that the CPU
// sends only the nearest few, and what keeps it honest is that the number is
// a console variable. A cell of a light grid is the next step and it is not
// this one.
//
// The decals are a flat array too, walked by every fragment the same way and
// for the same reason, and before anything is lit: a fragment inside a
// decal's box takes the decal's picture into its own color, normal,
// roughness and how metal it is, and is then lit as whatever it has become.
//
// The normal a pixel is shaded with is the geometry's, turned by whatever the
// normal map says. The frame that turn happens in is built per vertex from the
// normal and the tangent the mesh carries, and its third axis is the cross of
// the two times the tangent's sign - which is what makes a mirrored unwrap come
// out the right way up.

struct Globals {
    view_projection: mat4x4<f32>,
    // Clip space back into the world, for the sky, which is the one thing
    // drawn here that starts from a pixel and asks which way it is looking.
    inverse_view_projection: mat4x4<f32>,
    // xyz is the direction the light travels; w is unused.
    light: vec4<f32>,
    // rgb is how lit a surface facing away from the light still is.
    ambient: vec4<f32>,
    // xyz is where the camera is; w is unused.
    eye: vec4<f32>,
    // xyz is the direction it looks in; w is unused.
    forward: vec4<f32>,
    // World space into each cascade's clip space, nearest slice first.
    light_view_projection: array<mat4x4<f32>, 4>,
    // The view depth each cascade stops at, in world units.
    splits: vec4<f32>,
    // How many world units one texel of each cascade covers.
    cascade_texels: vec4<f32>,
    // x is one texel in map coordinates, y is unused, z is whether shadows are
    // on at all, w is whether to color every pixel by the cascade it read.
    shadow: vec4<f32>,
    // Where each cascade's map sits in the atlas, nearest slice first.
    cascade_tiles: array<Tile, 4>,
    // rgb is the color a distant surface fades towards; w is how quickly it
    // does, per unit of distance. A w of nought is no fog, and the arithmetic
    // below says so without a branch.
    fog: vec4<f32>,
    // rgb is the color straight up; w is whether a sky is drawn at all.
    sky_zenith: vec4<f32>,
    // rgb is the color at eye level; w is unused.
    sky_horizon: vec4<f32>,
    // rgb is the color straight down; w is unused.
    sky_ground: vec4<f32>,
    // x is how many of the lamps below are real and y how many of the decals;
    // the rest is unused.
    counts: vec4<u32>,
    // The local lights, nearest first. Everything from `counts.x` up is
    // whatever was in the buffer last frame and is never read.
    lamps: array<Lamp, MAX_LAMPS>,
    // The decals, in the order they are painted. Everything from `counts.y`
    // up is never read.
    decals: array<Paint, MAX_DECALS>,
};

// Where one shadow map sits in the atlas.
//
// Matched by `colby_engine::shadow::Tile`. A cascade's tile is a whole layer -
// origin nought, scale one, bounds nought to one - which makes the arithmetic
// below the identity for it, and that is why the atlas cost the sun's shadows
// no pixel at all.
struct Tile {
    // The rectangle a tap is held inside: min u, min v, max u, max v.
    bounds: vec4<f32>,
    // Where it starts, how much of a layer's side it covers, and which layer.
    place: vec4<f32>,
};

// How many local lights one frame may carry.
//
// Matched by `colby_engine::scene::MAX_LAMPS`, and the two have to agree: this
// sizes the uniform and that fills it.
const MAX_LAMPS: u32 = 32u;

// One point or cone, packed into three vectors.
//
// The kind is not a field, and that is the point: a cone's falloff is
// `saturate(cos * scale + offset)`, and a point is that same line with a scale
// of nought and an offset of one - which answers one everywhere and costs the
// loop no branch at all. Filament's packing, and bevy's.
struct Lamp {
    // xyz is where it is in the world; w is how far it reaches.
    position_range: vec4<f32>,
    // rgb is its color times its intensity; w is the cone's scale.
    color: vec4<f32>,
    // xyz is the way a cone points, which is the entity's own -z; w is the
    // cone's offset.
    direction: vec4<f32>,
};

// How many decals one frame may carry.
//
// Matched by `colby_engine::decal::MAX_DECALS`, and the two have to agree for
// the lamps' reason: this sizes the uniform and that fills it.
const MAX_DECALS: u32 = 32u;

// The bit in an instance's flags that says decals leave it alone.
const UNDECALED: u32 = 1u;

// One decal, packed into seven vectors.
struct Paint {
    // The world into the box's own space, a row an axis: a point's place
    // along one is the dot of xyz with it plus w, and the box is where all
    // three places are within a half of nought.
    rows: array<vec4<f32>, 3>,
    // Where the color picture is in the atlas, as u, v, width and height,
    // or all nought for a decal that throws its tint alone.
    color: vec4<f32>,
    // Where the normal map is, the same way, or all nought for none.
    normal: vec4<f32>,
    // rgb is the color; a is the opacity.
    tint: vec4<f32>,
    // x is metallic, y roughness, z how much it fades on a turned surface.
    surface: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

// Every picture a decal throws, in one texture read two ways: as colors,
// through a view that decodes sRGB, and as numbers for the normal maps.
@group(0) @binding(1) var decal_colors: texture_2d<f32>;
@group(0) @binding(2) var decal_numbers: texture_2d<f32>;
@group(0) @binding(3) var decal_sampler: sampler;

@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var surface_sampler: sampler;
// Sampled as numbers rather than as a color: the compiler stores it in a linear
// layout so that the GPU does not bend the directions on the way in.
@group(1) @binding(2) var normal_map: texture_2d<f32>;

// One layer per cascade. The comparison sampler answers "is this point behind
// what the light saw" rather than handing back a depth, and blends the answers
// rather than the depths - which is why one tap is already soft and why
// averaging depths here would be meaningless.
@group(2) @binding(0) var shadow_maps: texture_depth_2d_array;
@group(2) @binding(1) var shadow_sampler: sampler_comparison;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Origin top left; the importer flips OBJ's bottom-up v on the way in.
    @location(2) uv: vec2<f32>,
    // xyz is the direction u grows in; w is +1 or -1, and says which way the
    // third axis of the frame turns.
    @location(3) tangent: vec4<f32>,
};

struct InstanceInput {
    // A model matrix, one column per location. wgsl has no matrix vertex
    // attribute, so it arrives as four vectors and is put back together here.
    @location(4) model_0: vec4<f32>,
    @location(5) model_1: vec4<f32>,
    @location(6) model_2: vec4<f32>,
    @location(7) model_3: vec4<f32>,
    // The material's base color times the entity's own tint.
    @location(8) tint: vec4<f32>,
    // x is metallic, y is roughness, zw is how often the texture repeats.
    @location(9) surface: vec4<f32>,
    // xyz is one over the square of the entity's scale, which is the whole of
    // the normal matrix for a transform that is a translation, a rotation and a
    // scale. w is unused.
    @location(10) normal_scale: vec4<f32>,
    // x is where this instance's joint matrices start in the buffer below and
    // y is how many there are. Zero and zero is a thing bones do not move; the
    // static entry point never reads those two. z is the entity's own flags,
    // which both entry points hand on to the fragment stage.
    @location(11) skin: vec4<u32>,
};

// One matrix per bone of every posed character in the frame, back to back.
//
// One buffer rather than a block per character, because a block per character
// is a bind group per character and that is the batching thrown away. An
// instance carries the offset of its own run instead.
@group(3) @binding(0) var<storage, read> joints: array<mat4x4<f32>>;

struct SkinInput {
    // Which bones move this vertex, as indices into its own run.
    @location(12) bones: vec4<u32>,
    // How much each pulls. Normalized on the way in, so these are fractions
    // rather than the bytes the file holds, and they add to one.
    @location(13) weights: vec4<f32>,
};

// The one matrix that carries a vertex from the shape it was modeled in to
// where its bones have put it.
//
// The four are added rather than picked between: a vertex on a shoulder is
// partly the arm's and partly the chest's, and the weighted sum of the two
// matrices is what makes the surface between them bend instead of tear.
//
// The bone index is clamped rather than trusted. The importer already refuses
// one past the end of its own skeleton, so this is about the run: an index
// that walked off it would read the next character's bones and fling the
// vertex across the map.
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

    // unrolled because a vec4 may not be indexed by a value only known at run
    // time, which is four lines rather than a loop and a temporary array.
    return joints[at + min(skin.bones.x, last)] * skin.weights.x
        + joints[at + min(skin.bones.y, last)] * skin.weights.y
        + joints[at + min(skin.bones.z, last)] * skin.weights.z
        + joints[at + min(skin.bones.w, last)] * skin.weights.w;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // rgb is the material's color times the entity's; a is the material's
    // opacity, which only the blended entry point reads.
    @location(1) tint: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) world_position: vec3<f32>,
    @location(4) surface: vec2<f32>,
    // xyz is the tangent in world space; w carries the sign through unchanged.
    @location(5) tangent: vec4<f32>,
    // The entity's own flags, the same across a whole triangle.
    @location(6) @interpolate(flat) flags: u32,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    return place(vertex, instance, model_of(instance));
}

// The same for geometry bones move: the vertex is carried into its pose first
// and everything after that is identical.
//
// A separate entry point rather than a branch, because the two read different
// vertex buffers and a pipeline's buffers are fixed when it is built. What is
// not duplicated is anything below this line.
@vertex
fn vertex_skinned(vertex: VertexInput, instance: InstanceInput, skin: SkinInput) -> VertexOutput {
    let posed = skinning(skin, instance.skin.x, instance.skin.y);

    return place(vertex, instance, model_of(instance) * posed);
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

// Everything both entry points do once the model matrix is settled.
fn place(vertex: VertexInput, instance: InstanceInput, model: mat4x4<f32>) -> VertexOutput {
    let world_position = model * vec4<f32>(vertex.position, 1.0);

    var output: VertexOutput;
    output.clip_position = globals.view_projection * world_position;
    // the real normal matrix, and it costs three multiplies: for a model matrix
    // that is T * R * S the matrix carrying normals is R * S^-1, and
    // mat3(model) is R * S, so dividing by the square of the scale first leaves
    // exactly R * S^-1. Under a uniform scale this is the old line times a
    // constant, which normalize removes; under a stretched one it is the
    // difference between lighting the surface and lighting a lie.
    let normal = vertex.normal * instance.normal_scale.xyz;
    output.normal = (model * vec4<f32>(normal, 0.0)).xyz;
    // a tangent lies *in* the surface rather than across it, so it travels like
    // a position does and takes the model matrix unmodified.
    output.tangent = vec4<f32>(
        (model * vec4<f32>(vertex.tangent.xyz, 0.0)).xyz,
        vertex.tangent.w,
    );
    output.tint = instance.tint;
    output.uv = vertex.uv * instance.surface.zw;
    output.world_position = world_position.xyz;
    output.surface = instance.surface.xy;
    output.flags = instance.skin.z;

    return output;
}

// The shading normal: the geometry's, turned by the map.
//
// Gram-Schmidt again, because interpolating two frames across a triangle leaves
// a tangent that is no longer square with the normal beside it. A material with
// no map samples the flat texel, whose direction is straight out, and comes
// back with the normal it started with - so mapped and unmapped go down the
// same path, and the one branch below is about geometry rather than about
// whether there is a map.
fn shading_normal(input: VertexOutput) -> vec3<f32> {
    let normal = normalize(input.normal);
    let leaning = input.tangent.xyz;
    let tangent = leaning - normal * dot(normal, leaning);

    // a mesh whose unwrap collapsed has no tangent to speak of. The importer
    // gives it any perpendicular direction rather than a zero, so this is the
    // second line of the same defense and costs one comparison.
    if dot(tangent, tangent) < 1.0e-12 {
        return normal;
    }

    let along_u = normalize(tangent);
    let along_v = cross(normal, along_u) * input.tangent.w;
    let sampled = textureSample(normal_map, surface_sampler, input.uv).xyz * 2.0 - 1.0;

    return normalize(
        along_u * sampled.x + along_v * sampled.y + normal * sampled.z,
    );
}

// Which cascade covers a point, by the same measure the slices were cut on.
//
// Unrolled rather than looped, and reading the splits by name rather than by a
// running index, because a vector indexed with a value only known at run time
// is a thing some backends would rather not do. Four is not a number worth a
// loop anyway.
//
// Returns 4 for a point past the shadow distance, which is not a cascade and is
// how the caller learns there is nothing to sample.
fn cascade_of(view_depth: f32) -> i32 {
    if (view_depth > globals.splits.w) {
        return 4;
    }

    var slice = 3;
    if (view_depth <= globals.splits.z) { slice = 2; }
    if (view_depth <= globals.splits.y) { slice = 1; }
    if (view_depth <= globals.splits.x) { slice = 0; }

    return slice;
}

// How many world units one texel of a cascade covers. Unrolled for the reason
// above.
fn cascade_texel(slice: i32) -> f32 {
    if (slice <= 0) { return globals.cascade_texels.x; }
    if (slice == 1) { return globals.cascade_texels.y; }
    if (slice == 2) { return globals.cascade_texels.z; }

    return globals.cascade_texels.w;
}

// How much of the light reaches a point: one is lit, zero is fully in shadow.
//
// The sample is pushed along the surface's own normal before it is projected,
// by more of a texel the further the surface leans away from the light. That is
// what stops a lit surface striping itself: one shadow texel covers more and
// more depth as the surface turns edge on, so the point being tested has to be
// lifted out of its own texel by about as much.
fn shadowing(world_position: vec3<f32>, normal: vec3<f32>, lean: f32, slice: i32) -> f32 {
    if (globals.shadow.z < 0.5 || slice >= 4) {
        return 1.0;
    }

    let push = cascade_texel(slice) * mix(2.0, 4.0, clamp(lean, 0.0, 1.0));
    let clip = globals.light_view_projection[slice] * vec4<f32>(world_position + normal * push, 1.0);
    let ndc = clip.xyz / clip.w;

    // in front of the light's near plane, which nothing in the world should be:
    // the box is pulled back behind every caster. Past the far plane is a point
    // the cascade does not reach, and both answer the same way.
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 1.0;
    }

    // clip space counts y upwards and a texture counts it down.
    let at = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    return gather(globals.cascade_tiles[slice], at, ndc.z);
}

// How much of one map's light reaches a point already projected into it.
//
// Nine taps, and the tile is what turns them into places in the atlas: a tap
// sits at `origin + at * scale`, moved by a whole atlas texel each way, and is
// then held inside the tile's own bounds so it cannot wander into the map next
// door. One texel of the atlas is one texel of every tile in it, whatever the
// tile's size, which is why there is one step here and not one per tile.
//
// **For a cascade this is `at + offset` and nothing else.** The origin is
// nought, the scale is one and the bounds are the layer's own, so the multiply
// and the add are exact whether or not they are folded together, and clamping
// to nought and one in front of a sampler that already clamps to the edge
// changes no tap. That is the whole reason the cascades could move into an
// atlas without a picture moving with them.
//
// @param tile - where the map sits
// @param at - where the point landed in it, nought to one
// @param depth - how far the point is, in the map's own depth range
fn gather(tile: Tile, at: vec2<f32>, depth: f32) -> f32 {
    let step = globals.shadow.x;
    let layer = i32(tile.place.w);

    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * step;
            let uv = clamp(
                tile.place.xy + at * tile.place.z + offset,
                tile.bounds.xy,
                tile.bounds.zw,
            );

            lit += textureSampleCompareLevel(shadow_maps, shadow_sampler, uv, layer, depth);
        }
    }

    return lit / 9.0;
}

// A color per cascade, for the console variable that paints them.
fn cascade_color(slice: i32) -> vec3<f32> {
    if (slice <= 0) { return vec3<f32>(1.0, 0.55, 0.55); }
    if (slice == 1) { return vec3<f32>(0.55, 1.0, 0.55); }
    if (slice == 2) { return vec3<f32>(0.55, 0.7, 1.0); }
    if (slice == 3) { return vec3<f32>(1.0, 0.95, 0.55); }

    return vec3<f32>(1.0);
}

// The smoothest a surface is drawn, whatever its material says.
//
// Not nought: a mirror's highlight from a sun or a lamp is a point, and no
// pixel can hold a point. And not less than this, for two reasons that come
// out in the same place. Near its peak the divisor in `distribution_ggx` is the
// roughness to the fourth plus one minus the square of a number near one, and
// a float near one is only known to a step of about six hundred-millionths: at
// this roughness the fourth power is some seventy of those steps and the
// highlight keeps its shape, where at 0.02 it would be under three and the
// highlight would be the rounding. And the sun is half a degree across, and a
// highlight this smooth is already narrower than the sun's own reflection, so
// nothing smoother would be any truer under it.
//
// Matched by `colby_core::abi::material::MIN_ROUGHNESS`, and a test says the
// two agree.
const MIN_ROUGHNESS: f32 = 0.045;

// The brightest a surface comes out, in the target's own units.
//
// The target holds half floats, which stop at 65504, and a value past that can
// arrive as an infinity: a curve then divides it by itself and leaves a black
// dot in the middle of the highlight, and the meter reads the frame as so
// bright that the exposure goes to its floor. Only a smooth surface gets near
// it, under the sun at a grazing angle or under a lamp close by; at
// `MIN_ROUGHNESS` the sun alone reaches hundreds of thousands. Two to the
// fifteenth rather than all of the range, because the frame goes on blending
// into the target after a surface is drawn, glass over it and sparks added
// onto it, and what lands on top needs room. Past a few thousand everything is
// white on the screen anyway, so the number shows only in how far a highlight
// blooms.
const HDR_CEILING: f32 = 32768.0;

// How much of the surface's microfacets point along the half vector.
// Trowbridge-Reitz, which everyone calls GGX.
//
// With `a` the roughness squared, written as
// `(a / ((1 - n.h^2) + (n.h a)^2))^2 / pi` and not as the shorter
// `a^2 / (pi (n.h^2 (a^2 - 1) + 1)^2)`, which is the same function. In the
// shorter one `a^2 - 1` is a number near one, and adding the one back throws
// away the low bits of the very `a^2` the peak is made of: at `MIN_ROUGHNESS`
// its peak comes out six parts in a thousand low. This way round the peak is
// `1 / (pi a^2)` to the last bit, and what rounding is left sits in
// `1 - n.h^2` alone.
//
// **Nothing here is floored.** With the roughness held to `MIN_ROUGHNESS` and
// `normal_dot_half` to no more than one, the divisor is never below `a^2`, four
// millionths. A floor of a ten-thousandth under the old divisor flattened every
// highlight smoother than a roughness of 0.27: at 0.1 the peak came out 1,
// where it is 3183, and the lobe kept three and a half percent of its light.
fn distribution_ggx(normal_dot_half: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let aside = 1.0 - normal_dot_half * normal_dot_half;
    let along = normal_dot_half * a;
    let k = a / (aside + along * along);

    return k * k / 3.14159265;
}

// How much of them shadow each other, Smith's height-correlated form, already
// divided by the 4 * n.l * n.v the specular term would otherwise need.
//
// Not floored either: `normal_dot_view` arrives at a ten-thousandth or more and
// `a` at `MIN_ROUGHNESS` squared or more, so `light` alone is never below two
// ten-millionths. The floor this used to have bit only where the sun and the
// eye both graze the surface, and there it made a rim darker than it is.
fn visibility_smith(normal_dot_view: f32, normal_dot_light: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let view = normal_dot_light * sqrt(normal_dot_view * normal_dot_view * (1.0 - a2) + a2);
    let light = normal_dot_view * sqrt(normal_dot_light * normal_dot_light * (1.0 - a2) + a2);

    return 0.5 / (view + light);
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

// What one light of any kind does to a surface, before its own color and
// before anything in the way of it.
//
// Pulled out of `shade` when the second kind of light arrived: the sun and a
// lamp differ in where the direction comes from and in what multiplies the
// result, and in nothing at all between those two points. The pi is the
// convention this shader already had - the diffuse term is divided by it and
// the whole is multiplied back - and it is kept so that a lamp of intensity
// one and the sun are the same brightness head on.
fn lit_by(
    normal: vec3<f32>,
    towards_eye: vec3<f32>,
    towards_light: vec3<f32>,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    roughness: f32,
    normal_dot_view: f32,
) -> vec3<f32> {
    let half_vector = normalize(towards_light + towards_eye);
    let normal_dot_light = max(dot(normal, towards_light), 0.0);
    // no more than one: two unit vectors can dot to a hair past it, and with
    // nothing floored in the distribution that hair would lift the peak.
    let normal_dot_half = clamp(dot(normal, half_vector), 0.0, 1.0);
    let view_dot_half = max(dot(towards_eye, half_vector), 0.0);

    let fresnel = fresnel_schlick(view_dot_half, f0);
    let specular = fresnel
        * distribution_ggx(normal_dot_half, roughness)
        * visibility_smith(normal_dot_view, normal_dot_light, roughness);
    let diffuse = (vec3<f32>(1.0) - fresnel) * diffuse_color / 3.14159265;

    return (diffuse + specular) * normal_dot_light * 3.14159265;
}

// How much of a lamp survives the distance to a point.
//
// An inverse square with a window closed smoothly at the range, which is
// Karis's and Filament's and what bevy ships: the plain inverse square never
// reaches zero, so a lamp with no window either lights the whole world by a
// millionth or ends in a visible ring where somebody clipped it. The fourth
// power falls off slowly at first and steeply at the edge, so the window is
// invisible where the light is bright and complete where it is not.
fn lamp_falloff(distance_square: f32, range_square: f32) -> f32 {
    let factor = distance_square / max(range_square, 0.0001);
    let smoothed = clamp(1.0 - factor * factor, 0.0, 1.0);

    return smoothed * smoothed / max(distance_square, 0.0001);
}

// Everything the local lights add at one point.
//
// The whole array is walked and the ones past `counts.x` are not there. A
// fragment outside a lamp's range leaves the loop early rather than
// multiplying by a zero it already knows about, which is worth doing because
// the branch is coherent - neighboring fragments are inside or outside the
// same sphere together.
fn lamps_at(
    world_position: vec3<f32>,
    normal: vec3<f32>,
    towards_eye: vec3<f32>,
    f0: vec3<f32>,
    diffuse_color: vec3<f32>,
    roughness: f32,
    normal_dot_view: f32,
) -> vec3<f32> {
    var total = vec3<f32>(0.0);
    let count = min(globals.counts.x, MAX_LAMPS);

    for (var index = 0u; index < count; index++) {
        let lamp = globals.lamps[index];
        let towards = lamp.position_range.xyz - world_position;
        let distance_square = dot(towards, towards);
        let range_square = lamp.position_range.w * lamp.position_range.w;

        if (distance_square >= range_square) {
            continue;
        }

        let towards_light = towards * inverseSqrt(max(distance_square, 1.0e-8));
        // a point light packs a scale of nought and an offset of one, so this
        // is one for it whatever the angle is. @ref `Lamp`.
        let along = dot(-lamp.direction.xyz, towards_light);
        let cone = clamp(along * lamp.color.w + lamp.direction.w, 0.0, 1.0);

        total += lit_by(
            normal,
            towards_eye,
            towards_light,
            f0,
            diffuse_color,
            roughness,
            normal_dot_view,
        )
            * lamp.color.rgb
            * lamp_falloff(distance_square, range_square)
            * cone
            * cone;
    }

    return total;
}

// How much alpha a texel needs before a masked surface draws it at all.
//
// A constant rather than a number on the material: moving the picture's own
// alpha does the same job, and this is the same half that alpha to coverage
// falls back to on hardware without it. @ref `colby_core::abi::material::Blend`.
const MASK_CUTOFF: f32 = 0.5;

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // the texture is sRGB, so this is already linear by the time it is a float.
    // A material with no image samples the one white texel and multiplies by
    // one, which is why there is no branch here.
    //
    // One, not the tint's own alpha: a surface this pipeline draws is solid
    // whatever the material says, which is what every renderer with an alpha
    // mode does with the number in its other modes.
    return vec4<f32>(shade(input, textureSample(albedo, surface_sampler, input.uv)), 1.0);
}

// The same, for a surface whose picture has holes in it.
//
// A separate entry point rather than a branch, and it is the argument the two
// vertex entry points make arriving from the other end of the pipeline: a
// `discard` anywhere in a fragment shader stops the hardware throwing a
// fragment away before it is shaded, whether the branch is taken or not. One
// shared shader would therefore cost every solid surface in the world the early
// depth test it is passing today.
@fragment
fn fragment_masked(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo, surface_sampler, input.uv);
    if (sampled.a < MASK_CUTOFF) {
        discard;
    }

    return vec4<f32>(shade(input, sampled), 1.0);
}

// And for a surface what is behind still shows through.
//
// The alpha is the picture's times the material's, so frosted glass is a
// picture with an alpha channel at a material of one and a whole pane fading
// out is a flat picture at a material that moves. The pipeline blends it over
// what is already there; the depth buffer is read and not written, and the pass
// this runs in is sorted far to near, both of which are the pipeline's doing
// rather than anything this file can see.
@fragment
fn fragment_blended(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo, surface_sampler, input.uv);

    return vec4<f32>(shade(input, sampled), sampled.a * input.tint.a);
}

// What one point of a surface is made of, before it is lit: what its own
// material says, and then whatever the decals over it painted.
struct Surface {
    color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
};

// A surface with every decal over this point painted onto it, in the order the
// frame carries them, so that a later decal covers an earlier one.
//
// The whole list is walked, the lamps' way, and a point outside a decal's box
// leaves at the first test. That branch is coherent for the lamps' reason:
// neighboring fragments are inside or outside the same box together.
fn painted(
    start: Surface,
    world_position: vec3<f32>,
    facing: vec3<f32>,
    across: vec3<f32>,
    down: vec3<f32>,
) -> Surface {
    var surface = start;
    let count = min(globals.counts.y, MAX_DECALS);

    for (var index = 0u; index < count; index++) {
        let decal = globals.decals[index];
        let local = vec3<f32>(
            dot(decal.rows[0].xyz, world_position) + decal.rows[0].w,
            dot(decal.rows[1].xyz, world_position) + decal.rows[1].w,
            dot(decal.rows[2].xyz, world_position) + decal.rows[2].w,
        );

        if (any(abs(local) > vec3<f32>(0.5))) {
            continue;
        }

        surface = painted_by(surface, decal, local, facing, across, down);
    }

    return surface;
}

// One decal painted onto a surface, at a point inside its box.
//
// How much lands is the picture's own alpha times the material's opacity,
// faded towards the two faces of the box the picture is thrown between, so it
// does not end in a hard line on something that pokes through one, and faded
// on a surface turned away from the way it is thrown. What lands takes the
// color, how metal it is and the roughness towards the decal's own, and turns
// the normal by the decal's map if it has one.
fn painted_by(
    start: Surface,
    decal: Paint,
    local: vec3<f32>,
    facing: vec3<f32>,
    across: vec3<f32>,
    down: vec3<f32>,
) -> Surface {
    var surface = start;

    // the picture's own place: +x is its right edge and +y its top, and a
    // texture counts v downwards from its top
    let uv = vec2<f32>(local.x + 0.5, 0.5 - local.y);
    // how fast that place moves from one pixel to the next, which is what
    // picks a level: the point's own movement, carried into the box
    let along_x = vec2<f32>(dot(decal.rows[0].xyz, across), -dot(decal.rows[1].xyz, across));
    let along_y = vec2<f32>(dot(decal.rows[0].xyz, down), -dot(decal.rows[1].xyz, down));

    var picture = decal.tint;
    if (any(decal.color.xy > vec2<f32>(0.0))) {
        picture *= textureSampleGrad(
            decal_colors,
            decal_sampler,
            decal.color.xy + uv * decal.color.zw,
            along_x * decal.color.zw,
            along_y * decal.color.zw,
        );
    }

    let depth = abs(local.z) * 2.0;
    let square = depth * depth;
    let edge = 1.0 - square * square * square * square;
    let turned = dot(facing, normalize(decal.rows[2].xyz)) * 0.5 + 0.5;
    let fade = decal.surface.z;
    let facing_fade = select(1.0, smoothstep(fade, 1.0, turned), fade > 0.0);
    let amount = clamp(picture.a * edge * facing_fade, 0.0, 1.0);

    surface.color = mix(surface.color, picture.rgb, amount);
    surface.metallic = mix(surface.metallic, decal.surface.x, amount);
    surface.roughness = mix(surface.roughness, decal.surface.y, amount);

    if (any(decal.normal.xy > vec2<f32>(0.0))) {
        let numbers = textureSampleGrad(
            decal_numbers,
            decal_sampler,
            decal.normal.xy + uv * decal.normal.zw,
            along_x * decal.normal.zw,
            along_y * decal.normal.zw,
        ).xyz * 2.0 - 1.0;
        let right = normalize(decal.rows[0].xyz);

        surface.normal = normalize(mix(surface.normal, bent(surface.normal, right, numbers), amount));
    }

    return surface;
}

// A normal map's direction, laid on a surface along a decal's own axes.
//
// The picture's right edge is the box's x pressed flat into the surface, and
// down the picture is the way v grows: the frame a cube's face gives its own
// map, so a map that reads right on a cube reads right thrown.
fn bent(normal: vec3<f32>, right: vec3<f32>, numbers: vec3<f32>) -> vec3<f32> {
    let flat = right - normal * dot(normal, right);

    // a surface the box's x points straight into has no right edge to lay the
    // map along. The guard `shading_normal` keeps, for a surface the decal is
    // edge on to, which a fade leaves nearly unpainted anyway.
    if (dot(flat, flat) < 1.0e-12) {
        return normal;
    }

    let along_u = normalize(flat);
    let along_v = -cross(normal, along_u);

    return normalize(along_u * numbers.x + along_v * numbers.y + normal * numbers.z);
}

// Everything all three entry points do once the albedo has been sampled.
//
// Returns the color alone. What goes in the alpha channel is the one thing the
// three disagree about, so it is theirs rather than this function's.
fn shade(input: VertexOutput, sampled: vec4<f32>) -> vec3<f32> {
    // how far the point moves from one pixel to the next, asked here and not
    // among the decals: a derivative wants every pixel of a quad asking it
    // together, and whether the decals are asked at all is up to the entity.
    let across = dpdx(input.world_position);
    let down = dpdy(input.world_position);

    var surface = Surface(
        input.tint.rgb * sampled.rgb,
        input.surface.x,
        input.surface.y,
        shading_normal(input),
    );

    if ((input.flags & UNDECALED) == 0u) {
        surface = painted(surface, input.world_position, normalize(input.normal), across, down);
    }

    let base_color = surface.color;

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    // held here rather than where the material is read, so that what a person
    // typed is what gets saved, and after the decals, which move it per pixel.
    // @ref `MIN_ROUGHNESS` for how smooth that lets a surface be.
    let roughness = clamp(surface.roughness, MIN_ROUGHNESS, 1.0);

    let normal = surface.normal;
    let towards_light = normalize(-globals.light.xyz);
    let towards_eye = normalize(globals.eye.xyz - input.world_position);

    let normal_dot_light = max(dot(normal, towards_light), 0.0);
    let normal_dot_view = max(dot(normal, towards_eye), 0.0001);

    // a dielectric reflects four percent head on and is white doing it; a metal
    // reflects its own color and has no diffuse term at all.
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);
    let diffuse_color = base_color * (1.0 - metallic);

    // how much of the *sun* this point can see. It multiplies the sun's term
    // and nothing else: what a shadow takes away is that light's own
    // contribution, the lamps below cast none at all, and the ambient stands in
    // for everything that reaches a surface by some other route.
    let view_depth = dot(input.world_position - globals.eye.xyz, globals.forward.xyz);
    let slice = cascade_of(view_depth);
    let reaching = shadowing(input.world_position, normal, 1.0 - normal_dot_light, slice);

    let direct = lit_by(
        normal,
        towards_eye,
        towards_light,
        f0,
        diffuse_color,
        roughness,
        normal_dot_view,
    ) * reaching
        + lamps_at(
            input.world_position,
            normal,
            towards_eye,
            f0,
            diffuse_color,
            roughness,
            normal_dot_view,
        );

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
    // @ref `HDR_CEILING`: past it a smooth highlight would not fit the target.
    let color = min(direct + indirect, vec3<f32>(HDR_CEILING));

    if (globals.shadow.w > 0.5) {
        return color * cascade_color(slice);
    }

    return fogged(color, input.world_position);
}

// A surface faded towards the fog by how far away it is.
//
// **Here rather than in a pass of its own**, which is what every engine that
// has both does: a post pass would have to read the depth buffer as a texture,
// which means either a second depth target or a copy, and it would fog a pane
// of glass by the depth of whatever is behind it rather than by its own.
// Godot calls `fog_process(vertex)` from inside its forward fragment stage for
// the same two reasons.
//
// `exp(-(d * density)^2)` rather than `exp(-d * density)`: the square leaves
// what is near alone and closes over the far distance, where a plain
// exponential greys the whole picture evenly and reads as a dirty lens. A
// density of nought makes this `exp(0)`, which is one, which is the surface
// untouched - so there is no branch here and no cost worth one.
//
// The sky is not fogged. Its depth is the far plane rather than a surface's,
// so any density at all would turn the whole background one flat color; Godot
// exposes that as `fog_sky_affect` and it is a knob for another day. A scene
// that wants a horizon that closes sets its sky's own horizon color to the
// fog's, which is what everybody did before there was a knob.
fn fogged(color: vec3<f32>, world_position: vec3<f32>) -> vec3<f32> {
    let away = length(world_position - globals.eye.xyz) * globals.fog.w;

    return mix(globals.fog.rgb, color, exp(-away * away));
}

// The sky: a full-screen triangle at the far plane, shaded by which way each
// pixel is looking.
//
// **A triangle rather than a quad, and no vertex buffer at all.** Three
// vertices covering the screen have no seam down the middle for the rasterizer
// to sample twice, and the positions are arithmetic on the vertex index, so the
// draw is `draw(0..3)` with nothing bound.
//
// z is one, which is the far plane under wgpu's zero-to-one depth range. The
// pipeline tests depth with `less-equal` and writes none, so the sky survives
// exactly where the cleared depth is still one - every pixel no wall covered -
// and is thrown away everywhere else before it is shaded.
struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Where this corner is in clip space, carried through so the fragment
    // stage can unproject it. The builtin position is in pixels by then.
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vertex_sky(@builtin(vertex_index) index: u32) -> SkyOutput {
    // (-1,-1), (3,-1), (-1,3): a triangle whose middle is the screen.
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;

    var output: SkyOutput;
    output.clip_position = vec4<f32>(x, y, 1.0, 1.0);
    output.ndc = vec2<f32>(x, y);

    return output;
}

@fragment
fn fragment_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    // the ray through this pixel: the near plane and the far plane unprojected,
    // and the direction between them. Doing it per pixel rather than
    // interpolating three corner rays is what keeps it right under a wide field
    // of view, where the corners and the middle disagree.
    let near = globals.inverse_view_projection * vec4<f32>(input.ndc, 0.0, 1.0);
    let far = globals.inverse_view_projection * vec4<f32>(input.ndc, 1.0, 1.0);
    let way = normalize(far.xyz / far.w - near.xyz / near.w);

    return vec4<f32>(sky_color(way), 1.0);
}

// The gradient in one direction.
//
// Two halves meeting at the horizon, each eased so that the band at eye level
// is a band rather than a line: a linear ramp from the zenith straight to the
// ground puts all of its change at the poles and none where anybody is looking.
// The square is what Godot's `sky_curve` and `ground_curve` are for, fixed here
// at the value that reads as a sky rather than left as a number to tune.
fn sky_color(way: vec3<f32>) -> vec3<f32> {
    let up = clamp(way.y, -1.0, 1.0);

    if (up >= 0.0) {
        let t = 1.0 - (1.0 - up) * (1.0 - up);

        return mix(globals.sky_horizon.rgb, globals.sky_zenith.rgb, t);
    }

    let t = 1.0 - (1.0 + up) * (1.0 + up);

    return mix(globals.sky_horizon.rgb, globals.sky_ground.rgb, t);
}
