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
// Two more fragment entry points draw nothing anybody sees. The pass before the
// scene runs the same two vertex entry points into a target of its own and
// writes down, for every pixel of the solid and the masked half, the normal the
// picture is about to be lit with and how rough the surface is there - four
// more pipelines, and no second copy of any of the arithmetic below. @ref
// `colby_engine::prepass`.
//
// Shading is metallic-roughness: Cook-Torrance specular with GGX, Smith
// visibility and Schlick's Fresnel, over a Lambert diffuse. One directional
// light with cascaded shadows, up to MAX_LAMPS local ones whose maps are tiles
// of the same atlas, and no image-based lighting - the ambient term stands in
// for everything the scene does not simulate, which is why it is a color and
// not a number.
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
    // Where each local light's map sits, in the order the tiles were handed
    // out: a cone takes one and a point takes six in a row.
    lamp_tiles: array<Tile, MAX_LOCAL_TILES>,
    // World space into each of their clip spaces.
    lamp_views: array<mat4x4<f32>, MAX_LOCAL_TILES>,
    // rgb is the color a distant surface fades towards; w is how quickly it
    // does, per unit of distance. A w of nought is no fog, and the arithmetic
    // below says so without a branch.
    fog: vec4<f32>,
    // rgb is the color straight up; w is whether a sky is drawn at all.
    sky_zenith: vec4<f32>,
    // rgb is the color at eye level; w is how many roughness levels the
    // environment holds, nought for a world that has none.
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
    // x is the first atlas tile its map is in, y is how many tiles it has, z
    // is how much of the world one texel of that map covers per unit of
    // distance, and w is unused. A count of nought is a lamp that throws no
    // shadow, which is the only thing the loop below has to test.
    shadow: vec4<f32>,
};

// How many local shadow maps the atlas holds.
//
// Matched by `colby_engine::shadow::LOCAL_TILES`, for the reason MAX_LAMPS is
// matched: this sizes the uniform and that fills it.
const MAX_LOCAL_TILES: u32 = 16u;

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

// The environment, in the frame's own group because there is no fifth group -
// the scene binds four and four is the floor a device has to offer - and
// because the sky is drawn by a pipeline whose layout declares this group and
// no other. Its mip chain is a roughness chain rather than a size chain, so
// level nought is the picture the sky is drawn out of and a rough surface reads
// further down it.
@group(0) @binding(4) var environment: texture_cube<f32>;
@group(0) @binding(5) var environment_sampler: sampler;

// What fraction of whatever it reflects a surface actually sends back: the
// lobe above, integrated over a whole hemisphere, as a table of two numbers
// over n.v and roughness. Beside the environment because the frame reads it and
// no draw changes it, and bound whether there is an environment or not - a flat
// ambient color goes through the same table a sky does.
@group(0) @binding(6) var split_sum: texture_2d<f32>;
@group(0) @binding(7) var split_sampler: sampler;

// How many texels the table above is on a side. Matched by
// `colby_engine::brdf::SIDE`, and a test says the two agree - a shader cannot
// ask a texture how big it is without giving up the level argument.
const SPLIT_SIDE: f32 = 64.0;

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

// What the environment sends back along one direction, at one roughness.
//
// The level is the roughness, because the file's mip chain was built that way:
// nought is the picture itself and the last level is the whole hemisphere
// averaged. Multiplying by `levels - 1` is the whole mapping, and the sampler
// filters between two levels as well as inside them - without that a ball whose
// roughness changes smoothly over it shows a band wherever the level steps.
fn reflected_radiance(way: vec3<f32>, roughness: f32) -> vec3<f32> {
    let last = max(globals.sky_horizon.w - 1.0, 0.0);

    return textureSampleLevel(
        environment,
        environment_sampler,
        way,
        clamp(roughness, 0.0, 1.0) * last,
    ).rgb;
}

// What the environment sends a surface from every direction at once.
//
// The last level of the roughness chain, read along the normal: at a roughness
// of one the filter above is the widest the cube holds, so what comes back is
// the hemisphere around that direction averaged rather than anything sharp.
// That is the stand-in for an irradiance map, and it costs one tap of a texture
// already bound rather than a second cube, a second binding and a second half
// of the compiler.
//
// It is an average radiance and not an irradiance - the filter divides by the
// weights it summed - which is the same quantity `globals.ambient.rgb` is, so
// the two sides of the branch below multiply the diffuse color by the same kind
// of number.
fn ambient_radiance(way: vec3<f32>) -> vec3<f32> {
    return reflected_radiance(way, 1.0);
}

// The same, for a whole hemisphere of incoming light rather than one direction.
//
// **This is the lobe above integrated, not a curve that looks like it.** What
// the table holds is `f0 * A + B`, the two halves of the split-sum: the
// reflectance is linear in `f0`, so the part that multiplies it and the part
// that does not are summed separately once, offline, and every material picks
// its own colors out of the same two numbers. The f90 the bias is multiplied by
// is one, which is what @ref `fresnel_schlick` uses a direction at a time.
//
// **And then what a single bounce loses is put back.** The integral above is
// one reflection off one microfacet, and a rough surface's facets bounce light
// off each other: at a roughness of one the single bounce keeps thirty-one
// percent of what arrives and the rest is simply gone, so a rough metal comes
// out two thirds too dark rather than the too bright it used to be. The
// compensation is the standard one - scale by `1 + f0 (1/(A+B) - 1)`, which is
// exactly enough to send a white surface's whole hemisphere back - and three of
// the six engines in the field apply it. It cannot exceed one for an `f0` that
// does not, because `(f0 A + B) / (A + B)` is at most one when `f0` is.
//
// The reason the table is read rather than a published two-term fit evaluated:
// the fit is of somebody else's shadowing term, and against a real table it is
// out by six hundredths on average and by a fifth at a middling roughness head
// on. @ref `colby_engine::brdf` for where these numbers come from.
fn ambient_brdf(normal_dot_view: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let asked = vec2<f32>(clamp(normal_dot_view, 0.0, 1.0), clamp(roughness, 0.0, 1.0));
    // the table's first texel holds the value at nought and its last the value
    // at one, so a coordinate has to be squeezed into the strip between their
    // middles. Without this the outer half texel of each axis is flat, and the
    // two places it is flattest - the eye square to the surface, and a
    // roughness of one - are the two this term is read at most.
    let place = (asked * (SPLIT_SIDE - 1.0) + 0.5) / SPLIT_SIDE;
    let table = textureSampleLevel(split_sum, split_sampler, place, 0.0).rg;
    let once = f0 * table.x + vec3<f32>(table.y);
    // what one bounce keeps, which is what the compensation divides by
    let kept = max(table.x + table.y, 1.0e-4);

    return once * (vec3<f32>(1.0) + f0 * (1.0 / kept - 1.0));
}

// What the body of a material gets, once the reflection off its face has taken
// its share.
//
// A surface cannot send back more than reaches it. Whatever fraction the
// ambient reflection carries away never reaches the pigment underneath, so what
// is left for the diffuse term is one minus it - the same division @ref
// `lit_by` has made a direction at a time since before any of this existed,
// now made for the hemisphere as well.
//
// The max is not for that arithmetic, which cannot exceed one for an `f0` that
// does not. It is for a material whose color is above one, which nothing stops
// a person naming: without it such a surface would come back with a *negative*
// diffuse term and a black ring where it turns away.
fn ambient_diffuse(diffuse_color: vec3<f32>, reflected: vec3<f32>) -> vec3<f32> {
    return diffuse_color * max(vec3<f32>(0.0), vec3<f32>(1.0) - reflected);
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
        let lean = 1.0 - clamp(dot(normal, towards_light), 0.0, 1.0);

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
            * cone
            * lamp_shadowing(lamp, world_position, normal, lean);
    }

    return total;
}

// How much of one lamp's light reaches a point: one is lit, zero is in shadow.
//
// **A point light is six flat views and a cone is one**, which is why the only
// thing this has to decide is which of them a point fell into: the largest
// component of the direction from the lamp picks the face, in the order the
// matrices were written - `+x -x +y -y +z -z`. A cone has one map and skips
// the pick entirely.
//
// The sample is pushed along the surface's own normal first, for the reason a
// cascade's is, and by the same two-to-four texels - except that a texel of a
// perspective map grows with distance, so the size of it is worked out here
// from the distance to the lamp rather than read out of a table.
//
// **The face is picked from the pushed point, not the plain one.** A point
// near a face's edge can be pushed across it, and picking the face first would
// then project it through the wrong matrix and read the wrong map.
//
// @param lamp - the light, which carries where its maps are
// @param world_position - the point being lit
// @param normal - the surface's normal there
// @param lean - how far the surface is turned away from the lamp, nought
// facing it and one edge on
fn lamp_shadowing(lamp: Lamp, world_position: vec3<f32>, normal: vec3<f32>, lean: f32) -> f32 {
    let count = u32(max(lamp.shadow.y, 0.0));
    if (globals.shadow.z < 0.5 || count == 0u) {
        return 1.0;
    }

    let first = u32(max(lamp.shadow.x, 0.0));
    let reach = length(world_position - lamp.position_range.xyz);
    let push = lamp.shadow.z * reach * mix(2.0, 4.0, clamp(lean, 0.0, 1.0));
    let at = world_position + normal * push;

    var face = 0u;
    if (count > 1u) {
        let away = at - lamp.position_range.xyz;
        let size = abs(away);

        if (size.x >= size.y && size.x >= size.z) {
            face = select(1u, 0u, away.x > 0.0);
        } else if (size.y >= size.z) {
            face = select(3u, 2u, away.y > 0.0);
        } else {
            face = select(5u, 4u, away.z > 0.0);
        }
    }

    let index = min(first + face, MAX_LOCAL_TILES - 1u);
    let clip = globals.lamp_views[index] * vec4<f32>(at, 1.0);

    // behind the map's own eye, which a push across a face edge can just about
    // manage at grazing angles. Nothing there is in this map's shadow.
    if (clip.w <= 0.0) {
        return 1.0;
    }

    let ndc = clip.xyz / clip.w;

    // in front of the near plane or past the far one. The far one is the
    // lamp's own reach, so a point past it was already outside the falloff.
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return 1.0;
    }

    // clip space counts y upwards and a texture counts it down.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    return gather(globals.lamp_tiles[index], uv, ndc.z);
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

// What the pass before the scene writes for a solid surface: the normal it is
// about to be lit with, and how rough it is where the lobe reads it.
//
// **The surface `shade` lights, not a cheaper cousin of it.** The normal map and
// every decal have turned the normal by the time it is written, because this
// asks the same function `shade` asks.
//
// The albedo is not sampled. Its color is the one thing about a surface this
// pass does not write, and a solid surface keeps its whole face whatever the
// picture says, so a white texel stands in and is multiplied into a color
// nobody reads.
@fragment
fn fragment_prepass(input: VertexOutput) -> @location(0) vec4<f32> {
    return prepared(surface_at(input, vec4<f32>(1.0)));
}

// The same for a surface whose picture has holes in it, which leaves a hole in
// the buffer wherever it leaves one in the picture.
//
// **One cutoff and no coverage.** The pass is one sample a pixel whatever the
// picture is drawn with, so the alpha to coverage the picture's own pipeline
// turns on at four samples has nothing to spread across here: a leaf's edge is
// hard in the buffer where it is soft in the picture.
@fragment
fn fragment_prepass_masked(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo, surface_sampler, input.uv);
    if (sampled.a < MASK_CUTOFF) {
        discard;
    }

    return prepared(surface_at(input, sampled));
}

// A surface as the pass before the scene stores it: xyz the normal, in the
// world and of unit length, and w the roughness the lobe reads.
//
// **The roughness is held above `MIN_ROUGHNESS` here as `shade` holds it**,
// which is what the lobe is evaluated at, and it is also what makes nought in
// that channel mean that nothing was drawn: the pass clears the target to
// nought, and no surface is ever smoother than that.
fn prepared(surface: Surface) -> vec4<f32> {
    return vec4<f32>(surface.normal, clamp(surface.roughness, MIN_ROUGHNESS, 1.0));
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

// What one point of a surface is once its own picture and every decal over it
// have had their say: the half of `shade` that comes before any light.
//
// **A function of its own because two passes ask it.** The scene lights what
// this returns, and the pass before the scene writes down its normal and its
// roughness for whatever reads them before anything is lit - and a normal read
// in place of the one the picture is shaded with has to be this arithmetic
// rather than a copy of it that the next decal change leaves behind.
fn surface_at(input: VertexOutput, sampled: vec4<f32>) -> Surface {
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

    return surface;
}

// Everything all three entry points do once the albedo has been sampled.
//
// Returns the color alone. What goes in the alpha channel is the one thing the
// three disagree about, so it is theirs rather than this function's.
fn shade(input: VertexOutput, sampled: vec4<f32>) -> vec3<f32> {
    let surface = surface_at(input, sampled);
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
    // contribution, each lamp below asks its own maps for its own answer, and
    // the ambient stands in for everything that reaches a surface by some
    // other route.
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

    // everything this renderer does not simulate, in one term.
    //
    // @note: the specular half of it matters more than it looks. A metal has no
    // diffuse term at all, so without this a gold cube under one light is black
    // everywhere the highlight is not - physically right, and it reads as a bug.
    //
    // **Two lines rather than one, and the branch is still the point.** Folding
    // them together by multiplying by a white cube would not do: `a * (b + c)`
    // and `a * b + a * c` are not the same float, and a world with no sky has
    // no environment to read a level of.
    //
    // **Both halves are lit by the same thing.** When a world names an
    // environment, the specular half reads the level its roughness names along
    // the reflected direction and the diffuse half reads the roughest level
    // along the normal; when it does not, both read the one ambient color. What
    // `globals.ambient.rgb` means is therefore "the sky, for a world that has
    // none" - the same meaning it has had for the specular half since the cube
    // arrived, now taken to its conclusion.
    //
    // **And the diffuse half is what the specular one did not take.** A surface
    // cannot send back more than reaches it: whatever fraction the reflection
    // above carries away is gone before anything reaches the body of the
    // material underneath, so the diffuse term is multiplied by one minus it.
    // Three of the six engines in the field do this, one does it behind a
    // switch that is off for its older materials, and one refuses - but the
    // argument that settles it here is closer to home: `lit_by` has divided the
    // direct diffuse by the same Fresnel since before any of this existed, and
    // an ambient term that did not was the odd one out in this file.
    //
    let ambient_specular = ambient_brdf(normal_dot_view, f0, roughness);
    let indirect_diffuse = ambient_diffuse(diffuse_color, ambient_specular);
    var indirect: vec3<f32>;

    if (globals.sky_horizon.w > 0.5) {
        let reflected = reflect(-towards_eye, normal);

        indirect = ambient_radiance(normal) * indirect_diffuse
            + reflected_radiance(reflected, roughness) * ambient_specular;
    } else {
        indirect = globals.ambient.rgb * (indirect_diffuse + ambient_specular);
    }
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

    // one texture, two jobs: the picture is level nought of the same chain a
    // reflection reads further down. There is no second binding and no second
    // file, because level nought is what the filter leaves alone.
    if (globals.sky_horizon.w > 0.5) {
        return vec4<f32>(reflected_radiance(way, 0.0), 1.0);
    }

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
