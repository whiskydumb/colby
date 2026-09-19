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
// of the same atlas, and one term for everything the scene does not simulate:
// the world's environment when its sky names one and its one ambient color
// when it does not, both through the same split-sum table, and both multiplied
// by how much of the sky the surface can see - which nothing a light sends is.
// Where a smooth surface's reflection found something on the picture, what it
// found takes the environment's place in that term.
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
// A vertex may carry paint: a color somebody gave it, which multiplies the
// surface's own before anything is lit, and an alpha, which multiplies the
// picture's wherever the picture's is read - the cutout's edge and the pane's
// see-through alike. A mesh nobody painted reads white and opaque at every
// vertex, and multiplying by one is the identity. It may carry a second set of
// coordinates too, which a material's occlusion and glow may be read from.
//
// A material is more than the four numbers an instance carries: a picture of
// how metal and how rough, one of how much light from everywhere reaches each
// point, one of where it gives off light and how much, a turn and an offset of
// its first set of coordinates, and whether it is lit at all. Those are a
// uniform and three pictures of the material's own group, and every one of
// them is the identity at a material's default - a white texel multiplies by
// one, a turn of nought is the coordinates to the bit, and nought given off
// adds nought - which is what leaves every picture drawn before them the same.
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

// How much of the sky each pixel sees, worked out before this pass from the
// pass before the scene: half the picture on each axis, r the share and g how
// far along the view it was worked out, or one texel that says all of the sky
// in a frame nobody asked for it in. Beside the table for the table's reason.
// @ref `colby_engine::occlusion`, and `occlusion_at` for how it is read.
@group(0) @binding(8) var occlusion: texture_2d<f32>;

// What each pixel's reflection found on the picture, worked out before this pass
// from the pass before the scene: the picture's size, rgb the light already
// multiplied by a and a how much of the reflection it stands for, or one texel
// of nothing in a frame nobody asked for them in. Beside the share for the
// share's reason. @ref `colby_engine::reflection`, and `lit_at` for how it is
// mixed.
@group(0) @binding(9) var reflections: texture_2d<f32>;

// The world's lightmap: what a bake kept of the light arriving at every still
// thing from everywhere at once, one picture for all of them, in the unit the
// ambient color is in - or one texel of nothing in a frame that reads none, which
// no entry point that draws such a frame samples. One level, read at it through
// a clamped, bilinear sampler. Beside the reflections for their reason: the
// frame reads it and no draw changes it. @ref `colby_engine::lightmap`, and
// `baked_at` for how it is read.
@group(0) @binding(10) var lightmap: texture_2d<f32>;
@group(0) @binding(11) var lightmap_sampler: sampler;

// Where each thing's light is on that picture: one entry for every thing this
// frame drew that a bake gave a place to, xy how much of the picture its sheet
// spans and zw where on the picture it starts, both as fractions of it. An
// instance carries the index of its own entry in `skin.w`. @ref `place`.
@group(0) @binding(12) var<storage, read> places: array<vec4<f32>>;

@group(1) @binding(0) var albedo: texture_2d<f32>;
@group(1) @binding(1) var surface_sampler: sampler;
// Sampled as numbers rather than as a color: the compiler stores it in a linear
// layout so that the GPU does not bend the directions on the way in.
@group(1) @binding(2) var normal_map: texture_2d<f32>;

// What a material says beyond the four numbers its instances carry.
//
// Matched by `colby_engine::scene::Finish`, and by the same block in
// `shadow.wgsl`, which reads the turn for a cutout's holes; a test holds the
// three to one layout.
struct Finish {
    // rgb the light the surface gives off, its color already times its
    // strength; a how much of the occlusion picture is applied.
    glow: vec4<f32>,
    // How the first set of coordinates is turned, as two rows: `u' = x u + y v`
    // and `v' = z u + w v`, which is cos, sin, -sin and cos of the angle.
    turn: vec4<f32>,
    // xy where the first set's pictures start, added after the turn; zw unused.
    shift: vec4<f32>,
    // x the bits below; the rest unused.
    flags: vec4<u32>,
};

// `Finish::flags`: drawn as its own color, with no light on it.
const UNLIT: u32 = 1u;

// `Finish::flags`: the occlusion picture is read from the second set.
const OCCLUSION_UV2: u32 = 2u;

// `Finish::flags`: the glow picture is read from the second set.
const GLOW_UV2: u32 = 4u;

// Seen by the vertex stage too, which turns and moves the coordinates by it.
// At ten rather than three, because three to nine are the reflections' and the
// air's, whose passes bind layouts of their own in this group's place.
@group(1) @binding(10) var<uniform> finish: Finish;

// How metal in blue and how rough in green, multiplying the instance's two
// numbers: numbers, like the normal map, and the white texel for a material
// with no picture, which multiplies both by one.
@group(1) @binding(11) var finish_map: texture_2d<f32>;

// How much of the light arriving from everywhere reaches each point, in red:
// numbers too.
@group(1) @binding(12) var occlusion_map: texture_2d<f32>;

// Where the surface gives off light: a color, multiplying `finish.glow`.
@group(1) @binding(13) var glow_map: texture_2d<f32>;

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
    // The color the vertex was painted, linear, as four fractions; white and
    // opaque where nobody painted it. From a buffer of its own, which is why
    // its location follows the skin's rather than the tangent's.
    @location(14) paint: vec4<f32>,
    // The second set of coordinates, from the same buffer: nought where
    // nobody laid one out.
    @location(15) uv2: vec2<f32>,
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
    // which both entry points hand on to the fragment stage, and w which entry
    // of `places` is where its light is - meaningless for a thing a bake gave
    // no place, which is drawn by an entry point that never reads it.
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
    // How much of the picture the vertex's paint leaves, which multiplies the
    // picture's own alpha wherever that is read. One where nobody painted it.
    @location(7) paint_alpha: f32,
    // The second set of coordinates, as the mesh laid it out: nothing moves it.
    @location(8) uv2: vec2<f32>,
    // The same carried onto the lightmap by the thing's own place on it,
    // meaningless for a thing with none. Centroid: at four samples a pixel
    // on a triangle's edge can have its middle off the triangle, and a
    // coordinate taken there reaches past the ring a bake filled round the
    // chart into the next place's; the picture is read at a level named rather
    // than found from derivatives, so nothing else minds.
    @location(9) @interpolate(perspective, centroid) lightmap: vec2<f32>,
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
    // the paint multiplies the color once a vertex, here, so that the
    // fragment stage multiplies nothing it did not multiply before: what it
    // reads is the tint, whatever went into it. The alpha is left out of this
    // on purpose - that channel is the material's opacity, which only glass
    // reads, where the paint's alpha is the picture's and a cutout reads it too.
    output.tint = vec4<f32>(instance.tint.rgb * vertex.paint.rgb, instance.tint.a);
    output.paint_alpha = vertex.paint.a;
    // scaled, turned, then moved, which is the exchange format's order
    output.uv = finish.shift.xy + turned(vertex.uv * instance.surface.zw, finish.turn);
    output.uv2 = vertex.uv2;
    // one multiply and one add from the thing's own sheet onto the one picture
    // every still thing's light is on
    let kept = places[instance.skin.w];
    output.lightmap = kept.zw + vertex.uv2 * kept.xy;
    output.world_position = world_position.xyz;
    output.surface = instance.surface.xy;
    output.flags = instance.skin.z;

    return output;
}

// A set of coordinates turned by a material's angle.
//
// **Two rows, not a matrix**: `u' = cos u + sin v` and `v' = -sin u + cos v`,
// which turns the picture clockwise on a surface whose coordinates start at the
// picture's top left. That is the sense of the exchange format's own example
// and of the exporter that writes these files, where the format's code block,
// read as the column-major matrix it is written as, turns the other way.
// Written as two sums so that nothing reads which way a matrix is laid out.
//
// At an angle of nought this is `1 * u + -0 * v` and `0 * u + 1 * v`, which
// are `u` and `v` to the bit whether or not a compiler fuses them.
//
// **The same text is in `shadow.wgsl`**, whose cutouts have to be turned the
// way their pictures are, and a test holds the two to one text.
fn turned(uv: vec2<f32>, turn: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(turn.x * uv.x + turn.y * uv.y, turn.z * uv.x + turn.w * uv.y);
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
    let landed = cascade_landing(world_position + normal * push, slice);

    if (!landed.inside) {
        return 1.0;
    }

    return gather(globals.cascade_tiles[landed.tile], landed.uv, landed.depth);
}

// Where a point lands in one cascade's map, and how far along its depth.
//
// **One function for both of the things that ask**, for `lamp_landing`'s
// reason: a surface asks with its point pushed off its own face, and the air
// with its point where it is. Where a point lands and what the map holds there
// are the same question either way, and two answers to it would be two shadows
// of one sun.
//
// @param at - the point, already moved wherever the asker moves it
// @param slice - which cascade, nought to three, @ref `cascade_of`
fn cascade_landing(at: vec3<f32>, slice: i32) -> Landing {
    let clip = globals.light_view_projection[slice] * vec4<f32>(at, 1.0);
    let ndc = clip.xyz / clip.w;

    // in front of the light's near plane, which nothing in the world should be:
    // the box is pulled back behind every caster. Past the far plane is a point
    // the cascade does not reach, and both answer the same way.
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return Landing(0u, vec2<f32>(0.0), 0.0, false);
    }

    // clip space counts y upwards and a texture counts it down.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    return Landing(u32(slice), uv, ndc.z, true);
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

// The same out of one tap where `gather` takes nine: what a point of the air
// asks, which is averaged along its ray and across the tile afterwards and has
// no use for the eight around it. @ref `air_lamp_shadowing`.
//
// @param tile - where the map sits
// @param at - where the point landed in it, nought to one
// @param depth - how far the point is, in the map's own depth range
fn tapped(tile: Tile, at: vec2<f32>, depth: f32) -> f32 {
    let uv = clamp(tile.place.xy + at * tile.place.z, tile.bounds.xy, tile.bounds.zw);

    return textureSampleCompareLevel(shadow_maps, shadow_sampler, uv, i32(tile.place.w), depth);
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

    let reach = length(world_position - lamp.position_range.xyz);
    let push = lamp.shadow.z * reach * mix(2.0, 4.0, clamp(lean, 0.0, 1.0));
    let landed = lamp_landing(lamp, world_position + normal * push);

    if (!landed.inside) {
        return 1.0;
    }

    return gather(globals.lamp_tiles[landed.tile], landed.uv, landed.depth);
}

// Where a point lands in a shadow map: which tile, where in it, and how far
// along the map's own depth.
struct Landing {
    // an index into the table the asker reads - a lamp's tiles for a lamp, the
    // cascades' for the sun
    tile: u32,
    uv: vec2<f32>,
    depth: f32,
    // false for a point no map of the lamp's can say anything about
    inside: bool,
}

// Which of a lamp's maps a point falls in, and where.
//
// **One function for both of the things that ask**: a surface asks with its
// point pushed off its own face, and the air asks with a point that has no face
// to push off. Which face of a point light a direction picks and how the tile
// is read are the same question either way, and two answers to it would be two
// shadows of one lamp.
//
// @param lamp - the light, which carries where its maps are and how many
// @param at - the point, already moved wherever the asker moves it
fn lamp_landing(lamp: Lamp, at: vec3<f32>) -> Landing {
    let count = u32(max(lamp.shadow.y, 0.0));
    let first = u32(max(lamp.shadow.x, 0.0));
    let outside = Landing(0u, vec2<f32>(0.0), 0.0, false);

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
        return outside;
    }

    let ndc = clip.xyz / clip.w;

    // in front of the near plane or past the far one. The far one is the
    // lamp's own reach, so a point past it was already outside the falloff.
    if (ndc.z <= 0.0 || ndc.z >= 1.0) {
        return outside;
    }

    // clip space counts y upwards and a texture counts it down.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

    return Landing(index, uv, ndc.z, true);
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
    let sampled = textureSample(albedo, surface_sampler, input.uv);

    return vec4<f32>(shade(input, sampled, seen(input), found_at(input.clip_position.xy)), 1.0);
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
    // before anything decides to throw the fragment away, for the sample's
    // reason: @ref `seen`
    let lit = seen(input);
    if (sampled.a * input.paint_alpha < MASK_CUTOFF) {
        discard;
    }

    return vec4<f32>(shade(input, sampled, lit, found_at(input.clip_position.xy)), 1.0);
}

// The solid half again, for a surface a bake gave a place: what the bake kept
// takes the place of the sky's light on its diffuse half.
//
// **An entry point of its own rather than a word the solid one branches on.**
// The arithmetic that reads a lightmap and the arithmetic that does not are two
// sums over the same terms, and a compiler handed both in one function may
// rewrite the second to share work with the first - a rewrite that is not the
// same float, and was measured moving a picture that reads no lightmap at all.
// Split this way, every entry point that draws a thing with no place hands the
// baked light in as a constant nought, the branch that reads it folds away when
// the shader is compiled, and what is left is the arithmetic every picture was
// drawn with before there were lightmaps. @ref `shade_baked`.
@fragment
fn fragment_baked(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo, surface_sampler, input.uv);

    return vec4<f32>(shade_baked(input, sampled, seen(input), found_at(input.clip_position.xy)), 1.0);
}

// And the half with holes in it, for a surface a bake gave a place, for the
// reason above.
@fragment
fn fragment_masked_baked(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo, surface_sampler, input.uv);
    // before anything decides to throw the fragment away, for the sample's
    // reason: @ref `seen`
    let lit = seen(input);
    if (sampled.a * input.paint_alpha < MASK_CUTOFF) {
        discard;
    }

    return vec4<f32>(shade_baked(input, sampled, lit, found_at(input.clip_position.xy)), 1.0);
}

// And for a surface what is behind still shows through.
//
// The alpha is the picture's times the material's, so frosted glass is a
// picture with an alpha channel at a material of one and a whole pane fading
// out is a flat picture at a material that moves. The pipeline blends it over
// what is already there; the depth buffer is read and not written, and the pass
// this runs in is sorted far to near, both of which are the pipeline's doing
// rather than anything this file can see.
//
// **All of the sky, whatever is around it, and no reflection found.** Nothing
// that blends is in the buffers the share and the reflections were worked out
// from, so what they hold at a pane's pixels is the share of whatever is behind
// the pane and what its reflection found, and reading either would be lighting
// one surface with another's corners and another's mirror.
@fragment
fn fragment_blended(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(albedo, surface_sampler, input.uv);

    return vec4<f32>(
        shade(input, sampled, 1.0, vec4<f32>(0.0)),
        sampled.a * input.paint_alpha * input.tint.a,
    );
}

// How much of the sky the point a fragment of the solid or the masked half is
// shading can see.
//
// A function of its own because both of those ask it before anything else, and
// the masked one before it decides to throw the fragment away: how fast the
// distance changes from one pixel to the next is a derivative, and a
// derivative wants every pixel of a quad asking it together.
//
// **The distance is the w the view's own matrix gives the point, and not the
// eye and the forward direction that `shade` finds its cascade with.** The two
// are the same number. Written the same way, one channel of one pixel of a
// fixture moved by a level at a strength of nought, with nothing hidden
// anywhere - what a compiler sharing the two and rounding them another way
// would do, and what reading the texel alone did not. Written this way the
// picture at nought is the picture from before this was read, to the bit.
fn seen(input: VertexOutput) -> f32 {
    let along_view = (globals.view_projection * vec4<f32>(input.world_position, 1.0)).w;

    return occlusion_at(input.clip_position.xy, along_view, fwidth(along_view));
}

// How much of the sky a pixel's surface sees, read out of the half-sized buffer.
//
// **Four texels, blended by where the pixel lies between them.** A texel stands
// for the pixel at twice its place, so a pixel at an even place on both axes
// reads its own texel and nothing else, and one at an odd place the mean of the
// two or four around it. Written `a + (b - a) * t` rather than as a mix, so that
// four texels of one come back one to the last bit: an open surface multiplies
// its light by exactly one.
//
// **A texel that is not on this surface is not read.** One whose distance
// along the view is further from the fragment's than the surface's own slope
// allows is read as whichever of the four is nearest the fragment's distance
// instead. Without that, a thing standing in front of a crease carries the
// crease's darkness round its edge; and at four samples a pixel, a fragment on
// an edge is shaded at the middle of a pixel where the buffer may hold
// whatever is behind it. A slope rather than a share of the distance: on a
// floor seen at a slant, two pixels next to each other legitimately differ by
// more than a hundredth of how far away they are.
//
// **Every coordinate is held inside what is bound**, because a texel read past
// the end of a texture comes back nought, and nought is the whole sky hidden.
// The one texel a frame that asked for nothing binds answers one for every
// pixel of the picture that way.
//
// @param pixel - the fragment's place on the picture, in pixels
// @param along_view - how far along the view the fragment is
// @param slope - how much that changes from one pixel to the next
fn occlusion_at(pixel: vec2<f32>, along_view: f32, slope: f32) -> f32 {
    let last = vec2<i32>(textureDimensions(occlusion)) - vec2<i32>(1);
    let at = vec2<i32>(pixel);
    let base = at / 2;
    let t = vec2<f32>(at - base * 2) * 0.5;

    var texels = array<vec4<f32>, 4>(
        textureLoad(occlusion, min(base, last), 0),
        textureLoad(occlusion, min(base + vec2<i32>(1, 0), last), 0),
        textureLoad(occlusion, min(base + vec2<i32>(0, 1), last), 0),
        textureLoad(occlusion, min(base + vec2<i32>(1, 1), last), 0),
    );
    let within = slope * 1.5 + along_view * 1.0e-3;

    var nearest = texels[0].r;
    var closest = abs(texels[0].g - along_view);

    for (var index = 1u; index < 4u; index++) {
        let off = abs(texels[index].g - along_view);

        if (off < closest) {
            closest = off;
            nearest = texels[index].r;
        }
    }

    var shares = array<f32, 4>();

    for (var index = 0u; index < 4u; index++) {
        let own = abs(texels[index].g - along_view) <= within;

        shares[index] = select(nearest, texels[index].r, own);
    }

    let upper = shares[0] + (shares[1] - shares[0]) * t.x;
    let lower = shares[2] + (shares[3] - shares[2]) * t.x;

    return upper + (lower - upper) * t.y;
}

// What a fragment's reflection found on the picture: the texel of its own pixel,
// light already multiplied by how much of the reflection it stands for.
//
// **One texel and no match on distance.** The buffer is the picture's size and
// was brought up to it on each pixel's own surface already, so at one sample a
// pixel a fragment reads exactly what was found for it. At four, a fragment on
// an edge is shaded at the middle of a pixel whose texel may be the surface
// behind it, and reads that; the engines in the field that mix a reflection
// into a forward pass read it at the pixel the same way.
//
// **The coordinate is held inside what is bound**, as the share's is, though for
// another reason: a read past the end of a texture may come back as nought with
// an alpha of one, which here would be all of a reflection found and none of the
// light it stands for. Held, the one texel a frame that asked for nothing binds
// says nothing was found for every pixel of the picture.
//
// @param pixel - the fragment's place on the picture, in pixels
fn found_at(pixel: vec2<f32>) -> vec4<f32> {
    let last = vec2<i32>(textureDimensions(reflections)) - vec2<i32>(1);

    return textureLoad(reflections, min(vec2<i32>(pixel), last), 0);
}

// What the pass before the scene writes for a solid surface: the normal it is
// about to be lit with, how rough it is where the lobe reads it, and what it is
// made of.
//
// **The surface `shade` lights, not a cheaper cousin of it.** The normal map and
// every decal have turned the normal by the time it is written, because this
// asks the same function `shade` asks.
//
// **The albedo is sampled**, because the color is written: a reflection that
// finds this surface lights it again from what the buffer holds, and a surface
// lit with a white texel where its picture is would come back the wrong color.
// A solid surface keeps its whole face whatever the picture's alpha says, as
// the scene's own entry point does.
@fragment
fn fragment_prepass(input: VertexOutput) -> Prepared {
    let sampled = textureSample(albedo, surface_sampler, input.uv);

    return prepared(surface_at(input, sampled));
}

// The same for a surface whose picture has holes in it, which leaves a hole in
// the buffer wherever it leaves one in the picture.
//
// **One cutoff and no coverage.** The pass is one sample a pixel whatever the
// picture is drawn with, so the alpha to coverage the picture's own pipeline
// turns on at four samples has nothing to spread across here: a leaf's edge is
// hard in the buffer where it is soft in the picture.
@fragment
fn fragment_prepass_masked(input: VertexOutput) -> Prepared {
    let sampled = textureSample(albedo, surface_sampler, input.uv);
    if (sampled.a * input.paint_alpha < MASK_CUTOFF) {
        discard;
    }

    return prepared(surface_at(input, sampled));
}

// What the pass before the scene writes for one pixel, into its two targets.
struct Prepared {
    // xyz the normal, in the world and of unit length; w the roughness.
    @location(0) surface: vec4<f32>,
    // rgb the color, which is the material's times the entity's times the
    // picture's with every decal painted over it; a how metal it is.
    @location(1) material: vec4<f32>,
};

// A surface as the pass before the scene stores it.
//
// **The roughness is held above `MIN_ROUGHNESS` here as `shade` holds it**,
// which is what the lobe is evaluated at, and it is also what makes nought in
// that channel mean that nothing was drawn: the pass clears the target to
// nought, and no surface is ever smoother than that. **How metal it is is held
// inside nought and one as `shade` holds it**, and the color is not held at
// all, for the same reason: what is written is what the light is worked out
// from.
fn prepared(surface: Surface) -> Prepared {
    return Prepared(
        vec4<f32>(surface.normal, clamp(surface.roughness, MIN_ROUGHNESS, 1.0)),
        vec4<f32>(surface.color, clamp(surface.metallic, 0.0, 1.0)),
    );
}

// The numbers the pass that follows reflections reads. Laid out the way
// `colby_engine::reflection::Tuning` writes them, and the way `reflection.wgsl`
// reads the same block.
struct Mirror {
    // World space into view space, a row an axis.
    view_x: vec4<f32>,
    view_y: vec4<f32>,
    view_z: vec4<f32>,
    // x and y how far the projection scales a view-space x and y at a distance
    // of one; z and w its `z_axis.z` and `w_axis.z`, which turn a stored depth
    // back into a distance.
    lens: vec4<f32>,
    // x and y the size of the whole target in pixels; w how much of what a
    // reflection finds the picture takes.
    size: vec4<f32>,
    // The rectangle of the target the picture is drawn into: x, y, width,
    // height, in pixels.
    rect: vec4<f32>,
};

// What that pass reads, in the group the scene binds a material in: the pass
// binds no material, and a group a pipeline declares has to be bound, so the
// group it has room in is this one. At bindings the material's three are not,
// so that the two layouts are never two names for one slot.
@group(1) @binding(3) var<uniform> mirror: Mirror;
@group(1) @binding(4) var prepared_depth: texture_depth_2d;
@group(1) @binding(5) var prepared_surfaces: texture_2d<f32>;
@group(1) @binding(6) var prepared_material: texture_2d<f32>;

// The roughness at and past which no reflection is followed.
//
// **0.6, the field's middle**: one engine stops at a half, three at 0.6 and one
// at 0.7. Past it a reflection is so wide that what the picture holds along it
// is no better an answer than the prefiltered sky, and every texel followed
// costs a march.
const MIRROR_CUTOFF: f32 = 0.6;

// Where what a reflection finds starts to fade towards the cutoff, so that a
// surface whose roughness crosses it does not show a line.
const MIRROR_FADE: f32 = 0.5;

// How many steps a reflection takes across the picture at most, and never more
// than one a pixel.
const MIRROR_STEPS: u32 = 64u;

// How many times a step that went behind something is halved to find where it
// crossed.
const MIRROR_HALVINGS: u32 = 5u;

// How far behind what the picture shows a ray may be, as a share of that
// thing's distance, and still have met it rather than passed behind it.
//
// The depth is a surface seen from the eye and says nothing about how thick
// anything is, so this is the one guess a march cannot do without. A share
// rather than a length, because a pixel covers more depth the further away it
// is.
const MIRROR_THICKNESS: f32 = 0.05;

// How far along its direction a reflection is followed before it is cut to
// the picture, in world units: far enough that the picture's edge or the near
// plane cuts it first.
const MIRROR_REACH: f32 = 1000.0;

// Which way one texel's reflection leaves, drawn from the surface's own lobe.
//
// **One direction a texel, one of nine across a three by three tile**, the
// occlusion's tile and for its reason: the pass after this one averages a
// rough surface's texel with the eight around it, and on a surface that is
// every one of the nine directions exactly once, so what is left is the error
// of nine directions and not a pattern. The nine are three turns about the
// normal times three bands of how far a facet leans, each in the middle of its
// band, drawn through the same distribution the environment was filtered with.
//
// A facet whose reflection would go under the surface sends nothing back that
// way, and the mirror direction stands in for it.
//
// @param normal - the surface's, of unit length
// @param towards_eye - the way the eye is from the surface, of unit length
// @param roughness - the surface's, held above `MIN_ROUGHNESS` already
// @param cell - which of the nine, @ref `tile_of`
fn lobe_way(normal: vec3<f32>, towards_eye: vec3<f32>, roughness: f32, cell: u32) -> vec3<f32> {
    let turn = (f32(cell % 3u) + 0.5) / 3.0;
    let band = (f32(cell / 3u) + 0.5) / 3.0;
    let a = roughness * roughness;
    let cosine = sqrt((1.0 - band) / (1.0 + (a * a - 1.0) * band));
    let sine = sqrt(max(1.0 - cosine * cosine, 0.0));
    let angle = turn * 6.28318531;

    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
    let right = normalize(cross(helper, normal));
    let up = cross(normal, right);
    let facet = normalize(right * (cos(angle) * sine) + up * (sin(angle) * sine) + normal * cosine);
    let way = reflect(-towards_eye, facet);

    if (dot(way, normal) <= 0.0) {
        return reflect(-towards_eye, normal);
    }

    return way;
}

// Where a texel is in the three by three tile, as one of nine: the occlusion's
// order, whose neighbors never follow each other.
//
// **Counted from the corner of the rectangle the picture is drawn into**, not
// from the target's: a picture drawn into the middle of a window by the tools
// around it draws every direction where the same picture drawn alone draws it.
fn tile_of(texel: vec2<i32>) -> u32 {
    return tile_at(texel, mirror.rect);
}

// Where a texel is in the three by three tile, counted from the corner of a
// rectangle of the target: `tile_of`'s arithmetic for any pass that draws into
// a rectangle of its own.
fn tile_at(texel: vec2<i32>, rect: vec4<f32>) -> u32 {
    var order = array<u32, 9>(0u, 5u, 7u, 6u, 1u, 3u, 4u, 8u, 2u);
    let corner = vec2<i32>(rect.xy) / 2;
    let at = vec2<u32>(max(texel - corner, vec2<i32>(0))) % vec2<u32>(3u);

    return order[at.y * 3u + at.x];
}

// How far along the view a stored depth is.
fn mirror_distance(stored: f32) -> f32 {
    return mirror.lens.w / (stored + mirror.lens.z);
}

// A point of the view, laid out on the picture: in pixels of the target.
fn on_picture(ndc: vec2<f32>) -> vec2<f32> {
    return mirror.rect.xy + (ndc * vec2<f32>(0.5, -0.5) + 0.5) * mirror.rect.zw;
}

// Where the surface at a pixel of the target is in the world.
fn unprojected(pixel: vec2<i32>, stored: f32) -> vec3<f32> {
    let share = (vec2<f32>(pixel) + 0.5 - mirror.rect.xy) / mirror.rect.zw;
    let ndc = vec2<f32>(share.x * 2.0 - 1.0, 1.0 - share.y * 2.0);
    let point = globals.inverse_view_projection * vec4<f32>(ndc, stored, 1.0);

    return point.xyz / point.w;
}

// How much of a segment of the picture lies inside the rectangle drawn into,
// as a share of it, with half a pixel to spare at every edge.
fn inside_share(origin: vec2<f32>, across: vec2<f32>) -> f32 {
    let low = mirror.rect.xy + 0.5;
    let high = mirror.rect.xy + mirror.rect.zw - 0.5;
    var share = 1.0;

    if (across.x > 0.0) {
        share = min(share, (high.x - origin.x) / across.x);
    } else if (across.x < 0.0) {
        share = min(share, (low.x - origin.x) / across.x);
    }

    if (across.y > 0.0) {
        share = min(share, (high.y - origin.y) / across.y);
    } else if (across.y < 0.0) {
        share = min(share, (low.y - origin.y) / across.y);
    }

    return max(share, 0.0);
}

// Whether a point of a ray, as a share of its way across the picture, lies
// behind what the picture shows there.
//
// **Only behind a surface that faces the ray.** A ray cannot pass behind a
// surface it is leaving the front of, and treating it as behind one is what a
// ray grazing a floor on its way to the foot of a wall does for a pixel or two
// wherever the depth at a pixel's middle is a hair nearer than the ray at its
// edge: it then stays behind, crosses into the wall already behind it, and
// finds nothing at the wall's foot. Nothing is behind a pixel nothing was drawn
// in, either.
//
// @param way - the ray's direction in the world, of unit length
fn gone_behind(origin: vec3<f32>, finish: vec3<f32>, share: f32, way: vec3<f32>) -> bool {
    let at = mix(origin, finish, share);
    let place = vec2<i32>(floor(on_picture(at.xy)));
    let stored = textureLoad(prepared_depth, place, 0);

    return stored < 1.0 && at.z > stored && dot(textureLoad(prepared_surfaces, place, 0).xyz, way) < 0.0;
}

// What a march along one ray found: the pixel of the thing it met, and whether
// it met anything.
struct Found {
    place: vec2<i32>,
    met: bool,
};

// Follows a ray across the picture until it meets something the picture shows.
//
// **Evenly in the picture, not in the world**, and the depth of the ray is
// interpolated as the depth the picture stores: a projection carries a line to
// a line, and along it a stored depth is linear in the picture where a
// distance is not. A step at most a pixel long, from a pixel and a half out.
//
// **A step that goes behind something is halved until the crossing is found**,
// and the crossing is kept only if the ray is behind the thing there by less
// than `MIRROR_THICKNESS` of its distance and the thing faces the ray. What
// fails either test is a ray passing behind a silhouette or a surface seeing
// itself, and the march goes on past it.
//
// @param here - where the ray leaves from
// @param way - which way, of unit length
fn marched(here: vec3<f32>, way: vec3<f32>) -> Found {
    let missed = Found(vec2<i32>(0), false);
    let start = globals.view_projection * vec4<f32>(here, 1.0);
    var end = globals.view_projection * vec4<f32>(here + way * MIRROR_REACH, 1.0);

    // a ray coming towards the eye stops just short of the near plane, which is
    // where a stored depth is nought
    if (end.z < 0.0) {
        end = mix(start, end, start.z / (start.z - end.z) * 0.999);
    }

    let origin = start.xyz / start.w;
    let finish = end.xyz / end.w;
    let start_pixel = on_picture(origin.xy);
    let across = on_picture(finish.xy) - start_pixel;
    let whole = length(across);
    let limit = inside_share(start_pixel, across);
    let length_inside = whole * limit;

    if (length_inside < 2.0) {
        return missed;
    }

    let first = 1.5 / whole;
    let steps = min(MIRROR_STEPS, u32(length_inside));
    var before = first;
    var was_behind = false;

    for (var step = 1u; step <= steps; step++) {
        let share = first + (limit - first) * f32(step) / f32(steps);
        let behind = gone_behind(origin, finish, share, way);

        if (behind && !was_behind) {
            var low = before;
            var high = share;

            for (var halving = 0u; halving < MIRROR_HALVINGS; halving++) {
                let middle = (low + high) * 0.5;

                if (gone_behind(origin, finish, middle, way)) {
                    high = middle;
                } else {
                    low = middle;
                }
            }

            let at = mix(origin, finish, high);
            let place = vec2<i32>(floor(on_picture(at.xy)));
            let stored = textureLoad(prepared_depth, place, 0);
            let held = textureLoad(prepared_surfaces, place, 0);
            let how_far = mirror_distance(stored);
            let gap = mirror_distance(at.z) - how_far;

            if (held.w > 0.0 && gap <= MIRROR_THICKNESS * how_far) {
                return Found(place, true);
            }
        }

        was_behind = behind;
        before = share;
    }

    return missed;
}

// What one texel of the picture's reflection finds on the picture: rgb the
// light the thing it met sends back towards it, and a how much of the
// reflection that stands for - one where something was met and nought where
// nothing was, both faded towards the roughness past which nothing is
// followed. Already multiplied, so an average of them is an average of light.
//
// **The thing met is lit here, not read off a picture.** Every screen-space
// reflection in the field reads the color the picture already has at the
// place a ray lands, and three of four read it off the frame before, because
// in a renderer that lights as it draws the picture for this frame does not
// exist yet. This one lights the place from what the pass before the scene
// wrote about it, with `lit_at` - the scene's own arithmetic, towards the point
// the ray left rather than towards the eye - so one frame is enough, nothing
// is a frame late, and a polished thing seen in a mirror shows what it shows
// from the mirror rather than what it shows the eye. What is not lit is what
// the buffers do not hold: anything blended, particles, the lines, and the
// reflections the thing met would itself show, for which the sky stands in -
// the thing is lit with nothing found, so what the last frame's reflections
// found never comes back into this one's. @ref `colby_engine::reflection`.
//
// **All of it at the whole strength, and a share of it at a share**: rgb and a
// alike, so that a reflection taken at half its strength is half the found
// light mixed over half of what the environment would have given.
@fragment
fn fragment_reflections(input: SkyOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let pixel = texel * 2;
    let stored = textureLoad(prepared_depth, pixel, 0);
    let held = textureLoad(prepared_surfaces, pixel, 0);
    let roughness = held.w;

    if (stored >= 1.0 || roughness <= 0.0 || roughness >= MIRROR_CUTOFF) {
        return vec4<f32>(0.0);
    }

    let here = unprojected(pixel, stored);
    let way = lobe_way(held.xyz, normalize(globals.eye.xyz - here), roughness, tile_of(texel));
    let found = marched(here, way);

    if (!found.met) {
        return vec4<f32>(0.0);
    }

    let place = found.place;
    let there = unprojected(place, textureLoad(prepared_depth, place, 0));
    let hit = textureLoad(prepared_surfaces, place, 0);
    let made = textureLoad(prepared_material, place, 0);
    // the pass before the scene has no room for the material's own occlusion
    // or for the light it gives off, so a thing seen in a mirror is lit as if
    // it had neither
    let surface = Surface(made.rgb, made.a, hit.w, hit.xyz, 1.0, vec3<f32>(0.0));
    let slice = cascade_of(dot(there - globals.eye.xyz, globals.forward.xyz));
    // the share of the sky the thing met sees, read at its own texel: the one
    // texel a frame that asks for no occlusion binds says all of it
    let last = vec2<i32>(textureDimensions(occlusion)) - vec2<i32>(1);
    let lit = textureLoad(occlusion, min(place / 2, last), 0).r;
    // and no baked light either: the pass before the scene has no room for
    // where a thing's light is on the lightmap, so what a mirror finds is lit
    // by the sky
    let unbaked = vec4<f32>(0.0);
    let light = lit_at(surface, there, normalize(here - there), slice, lit, vec4<f32>(0.0), unbaked);
    let fade = 1.0 - smoothstep(MIRROR_FADE, MIRROR_CUTOFF, roughness);

    return vec4<f32>(light * fade, fade) * mirror.size.w;
}

// The numbers the passes that light the air read. Laid out the way
// `colby_engine::haze::Tuning` writes them, and the way `haze.wgsl` reads the
// same block.
struct Air {
    // x how much of the light crossing a unit of air the air scatters; y how
    // far from the eye the air goes on; z and w the projection's `z_axis.z`
    // and `w_axis.z`, which turn a stored depth back into a distance.
    medium: vec4<f32>,
    // x and y the size of the whole target in pixels; z and w unused.
    size: vec4<f32>,
    // The rectangle of the target the picture is drawn into: x, y, width,
    // height, in pixels.
    rect: vec4<f32>,
};

// What those passes read, in the group the scene binds a material in: neither
// binds a material, and a group a pipeline declares has to be bound. At
// bindings neither the material's three nor the reflections' four are, so that
// no two layouts are two names for one slot.
@group(1) @binding(7) var<uniform> air: Air;

// The depth the scene wrote, one sample a pixel whatever it was drawn with.
@group(1) @binding(8) var air_depth: texture_depth_2d;

// What the march found, averaged, at half the picture on each axis. Read by the
// pass that puts it over the picture, and bound beside the march as well, which
// reads it nowhere: one group for both.
@group(1) @binding(9) var air_found: texture_2d<f32>;

// How many places a lamp's light is sampled at along the stretch of a ray the
// lamp can reach.
//
// **Sixteen, measured rather than chosen**, with the engine at seven-twenty
// against the same march at five hundred and twelve places and nothing
// averaged: a point lamp's glow was left with 39 pixels past two levels at
// sixteen places spread by the angle below, 234 at eight and 24 at thirty-two,
// and with 829 at sixteen spread evenly in distance. Sixteen is the number of
// the one engine in the field with this shape, too.
const AIR_STEPS: u32 = 16u;

// How many lamps one texel's ray is followed through, the nearest first.
//
// A frame carries up to thirty-two and a ray through the middle of a lit room
// can cross the reach of all of them; this is what keeps a texel at sixty-four
// taps of a shadow map at the most, the reflections' count.
const AIR_LAMPS: u32 = 4u;

// How much more of what the air scatters carries on the way it was going than
// comes back: the asymmetry of Henyey and Greenstein's lobe. Two tenths, where
// the two engines in the field that start from a number start.
const AIR_ASYMMETRY: f32 = 0.2;

// The nearest a ray is counted as passing a lamp, in world units.
//
// A ray through a point light's very middle meets an inverse square with
// nothing under it, and the sum along the ray has no finite answer; the surface
// shading holds the same distance off at a centimeter.
const AIR_NEAREST: f32 = 0.01;

// How far a texel's distance may be from a pixel's, as a share of the pixel's,
// and still stand for the air in front of the same surface.
const AIR_PLANE: f32 = 0.05;

// One ray of the air: where it starts, which way it goes, and how far.
struct AirRay {
    origin: vec3<f32>,
    way: vec3<f32>,
    // to what the pixel shows, or to where the air ends, whichever is nearer
    far: f32,
}

// How far along the view a stored depth is.
fn air_distance(stored: f32) -> f32 {
    return air.medium.w / (stored + air.medium.z);
}

// The ray through the middle of a pixel of the target, out to whatever the
// depth says is there - the far plane, where nothing was drawn.
fn air_ray(pixel: vec2<i32>) -> AirRay {
    let stored = textureLoad(air_depth, pixel, 0);
    let share = (vec2<f32>(pixel) + 0.5 - air.rect.xy) / air.rect.zw;
    let ndc = vec2<f32>(share.x * 2.0 - 1.0, 1.0 - share.y * 2.0);
    let point = globals.inverse_view_projection * vec4<f32>(ndc, stored, 1.0);
    let seen = point.xyz / point.w - globals.eye.xyz;
    let distance = length(seen);

    return AirRay(globals.eye.xyz, seen / max(distance, 1.0e-6), min(distance, air.medium.y));
}

// The stretch of a ray a lamp can light: inside its reach, inside a cone's
// cone, and short of the ray's end. Nought long for a lamp the ray misses.
fn air_span(lamp: Lamp, ray: AirRay) -> vec2<f32> {
    let offset = ray.origin - lamp.position_range.xyz;
    let b = dot(ray.way, offset);
    let c = dot(offset, offset) - lamp.position_range.w * lamp.position_range.w;
    let discriminant = b * b - c;

    if (discriminant <= 0.0) {
        return vec2<f32>(0.0);
    }

    let root = sqrt(discriminant);
    let near = max(-b - root, 0.0);
    let far = min(-b + root, ray.far);

    if (far <= near) {
        return vec2<f32>(0.0);
    }

    // a point light packs a scale of nought, and its reach is all there is
    if (lamp.color.w == 0.0) {
        return vec2<f32>(near, far);
    }

    return air_cone(lamp, ray, vec2<f32>(near, far));
}

// The part of a stretch of a ray inside a cone's lit half.
//
// **A line meets the lit half of a cone, which is convex, in one piece at
// most**, so of the three pieces the two places the line crosses the cone's
// surface cut the stretch into, the one whose middle is inside is the answer,
// and a line that never crosses is inside or outside all the way along. The
// edge is where the cone's falloff reaches nought, out of the two numbers it is
// packed as: a lamp's light past it is none, so the air there needs no step.
fn air_cone(lamp: Lamp, ray: AirRay, span: vec2<f32>) -> vec2<f32> {
    let axis = lamp.direction.xyz;
    let edge = -lamp.direction.w / lamp.color.w;
    let edge_square = edge * edge;
    let offset = ray.origin - lamp.position_range.xyz;
    let way_along = dot(ray.way, axis);
    let offset_along = dot(offset, axis);
    let qa = way_along * way_along - edge_square;
    let qb = 2.0 * (way_along * offset_along - dot(ray.way, offset) * edge_square);
    let qc = offset_along * offset_along - dot(offset, offset) * edge_square;
    let discriminant = qb * qb - 4.0 * qa * qc;
    var cuts = array<f32, 4>(span.x, span.x, span.y, span.y);

    if (discriminant > 0.0 && abs(qa) > 1.0e-8) {
        let root = sqrt(discriminant);
        let one = (-qb - root) / (2.0 * qa);
        let other = (-qb + root) / (2.0 * qa);

        cuts[1] = clamp(min(one, other), span.x, span.y);
        cuts[2] = clamp(max(one, other), span.x, span.y);
    }

    for (var piece = 0u; piece < 3u; piece++) {
        let start = cuts[piece];
        let end = cuts[piece + 1u];

        if (end <= start) {
            continue;
        }

        let middle = offset + ray.way * ((start + end) * 0.5);

        if (dot(middle, axis) > edge * length(middle)) {
            return vec2<f32>(start, end);
        }
    }

    return vec2<f32>(0.0);
}

// How much of the light a unit of air scatters goes one way, per unit of solid
// angle: Henyey and Greenstein's lobe, which sums to one over every way there
// is.
//
// @param cosine - between the way the light was going and the way it leaves
fn scattered(cosine: f32) -> f32 {
    let g = AIR_ASYMMETRY;
    let denominator = 1.0 + g * g - 2.0 * g * cosine;

    return (1.0 - g * g) / (4.0 * 3.14159265 * denominator * sqrt(denominator));
}

// How much of one lamp's light reaches a point of the air: one tap of its map,
// where a surface takes nine.
//
// **One tap rather than nine**: a point of the air is averaged with its
// neighbors along the ray and across the tile afterwards, which is what the
// eight around the middle tap buy a surface. The point is moved a little way
// towards the lamp first, for the reason a surface is pushed off its own face:
// a point of the air just in front of a wall would read the wall's own depth
// and put itself in the wall's shadow.
fn air_lamp_shadowing(lamp: Lamp, at: vec3<f32>) -> f32 {
    let count = u32(max(lamp.shadow.y, 0.0));
    if (globals.shadow.z < 0.5 || count == 0u) {
        return 1.0;
    }

    let towards = lamp.position_range.xyz - at;
    let reach = length(towards);
    let push = lamp.shadow.z * reach * 2.0;
    let landed = lamp_landing(lamp, at + towards * (push / max(reach, 1.0e-6)));

    if (!landed.inside) {
        return 1.0;
    }

    return tapped(globals.lamp_tiles[landed.tile], landed.uv, landed.depth);
}

// The light one lamp sends towards the eye out of one stretch of a ray.
//
// **The places sampled are spread evenly in the angle the stretch covers as
// seen from the lamp**, not evenly in distance. Each stands for as much of the
// lamp's view as the next, which crowds them where the ray passes the lamp -
// where an inverse square puts nearly all of its light - and the weight each
// carries is that inverse square turned round, so a ray passing close to a lamp
// adds up what is there rather than stepping over it. It is how the field's
// path tracers sample a point light in a medium, and for the same reason.
//
// What each place adds is the lamp's own light the way a surface would get it -
// its falloff, a cone's cone and its shadow - scattered towards the eye by the
// lobe above, and dimmed by the air between the place and the eye. The pi is
// the one `lit_by` multiplies back, so that the air in front of a lamp and a
// white wall in front of it are lit by the same lamp.
//
// @param offset - where in its step each place sits, nought to one, which is
// what the tile spreads
fn air_lamp(lamp: Lamp, ray: AirRay, span: vec2<f32>, offset: f32) -> vec3<f32> {
    let to_lamp = lamp.position_range.xyz - ray.origin;
    let closest = dot(to_lamp, ray.way);
    let apart = max(length(to_lamp - ray.way * closest), AIR_NEAREST);
    let first = atan((span.x - closest) / apart);
    let last = atan((span.y - closest) / apart);
    let range_square = lamp.position_range.w * lamp.position_range.w;
    var total = 0.0;

    for (var step = 0u; step < AIR_STEPS; step++) {
        let angle = mix(first, last, (f32(step) + offset) / f32(AIR_STEPS));
        let along = closest + apart * tan(angle);
        let at = ray.origin + ray.way * along;
        let leaving = at - lamp.position_range.xyz;
        let distance_square = dot(leaving, leaving);
        let leaving_way = leaving * inverseSqrt(max(distance_square, 1.0e-8));
        // a point light packs a scale of nought and an offset of one
        let cone = clamp(dot(lamp.direction.xyz, leaving_way) * lamp.color.w + lamp.direction.w, 0.0, 1.0);
        let spread = apart * apart + (along - closest) * (along - closest);

        total += lamp_falloff(distance_square, range_square)
            * cone
            * cone
            * scattered(dot(leaving_way, -ray.way))
            * exp(-air.medium.x * along)
            * air_lamp_shadowing(lamp, at)
            * spread;
    }

    return lamp.color.rgb * (total * air.medium.x * (last - first) / (f32(AIR_STEPS) * apart) * 3.14159265);
}

// How many places the sun's light is sampled at along a texel's ray.
//
// **Sixteen, measured rather than chosen**, with the engine at seven-twenty
// against the same march at five hundred and twelve places and nothing
// averaged, on a beam of sun crossing a room through a window: 9,000 pixels
// past two levels at sixteen, 39,097 at eight, 4,326 at twenty-four and 2,422 at
// thirty-two, where every eight places cost the frame some twenty-three
// microseconds. A number of its own rather than `AIR_STEPS`, because the two
// estimates are not one: a lamp's places are spread by angle over the stretch
// it reaches, and these evenly over the whole ray.
const AIR_SUN_STEPS: u32 = 16u;

// How much of the sun's light reaches a point of the air: one tap of the map of
// the cascade the point's own distance along the view falls in.
//
// **The cascade is the point's and not the ray's**: a ray crosses every slice
// between the eye and whatever it ends on, and a point is inside the box of the
// slice its own view depth picks and no other - the rule a surface follows. A
// point past the shadow distance is in no slice, and gets all of the light, as
// a surface there does.
//
// **Not pushed towards the sun**, where a lamp's air is pushed towards the lamp.
// The cascades are drawn with a slope-scaled bias and a lamp's maps with none,
// and the bias already lifts a lit surface's depth away from the sun by as much
// as the surface is turned from it, which is what keeps the air just in front of
// the surface out of its shadow. A push on top of that only carries a point in
// the shadow just under an edge the sun grazes into the edge and past its
// lifted depth. Measured against a second answer on a window: two texels of
// push put 23 pixels of the picture under its sill past two levels, the worst
// of them at 19, where no push leaves one at 3.
//
// @param at - the point
fn air_sun_shadowing(at: vec3<f32>) -> f32 {
    let slice = cascade_of(dot(at - globals.eye.xyz, globals.forward.xyz));

    if (globals.shadow.z < 0.5 || slice >= 4) {
        return 1.0;
    }

    let landed = cascade_landing(at, slice);

    if (!landed.inside) {
        return 1.0;
    }

    return tapped(globals.cascade_tiles[landed.tile], landed.uv, landed.depth);
}

// The light the sun sends towards the eye out of the air along one ray.
//
// **The places are spread evenly along the whole ray**, not towards the eye:
// the sun has no reach to cut a stretch out of and no inverse square to crowd
// them at, and what takes its light away is a shadow that can fall anywhere
// along the ray. Spread instead by how much of the air's light is left at each,
// which crowds them near the eye, they left 42,726 pixels past two levels on the
// same beam where these leave 9,000 - and worse the thicker the air, because a
// beam crosses the middle of a room and not the air at the eye.
//
// Each place adds its shadow dimmed by the air between it and the eye. The lobe
// is one number for the whole ray, because the sun's light travels the same way
// everywhere, and the pi is the one `lit_by` multiplies back, as a lamp's is: a
// white wall facing the sun and the air in front of it are lit by the same sun.
//
// @param offset - where in its step each place sits, nought to one, which is
// what the tile spreads
fn air_sun(ray: AirRay, offset: f32) -> vec3<f32> {
    let travel = normalize(globals.light.xyz);
    var total = 0.0;

    for (var step = 0u; step < AIR_SUN_STEPS; step++) {
        let along = ray.far * ((f32(step) + offset) / f32(AIR_SUN_STEPS));

        total += air_sun_shadowing(ray.origin + ray.way * along) * exp(-air.medium.x * along);
    }

    return vec3<f32>(scattered(dot(travel, -ray.way)) * air.medium.x * ray.far * total / f32(AIR_SUN_STEPS) * 3.14159265);
}

// The light the air along one texel's ray sends towards the eye: the sun's, and
// every lamp's the frame carries whose light the ray crosses, up to `AIR_LAMPS`
// of them, the nearest first - the sun is none of the four. rgb the light and
// nothing in the fourth.
//
// **One ray a texel**, through the middle of the pixel the texel stands for, out
// to what the depth after the scene says is there. Where along its steps a
// texel's samples sit is one of nine places over the occlusion's three by three
// tile, and the pass after this one averages each texel with its neighbors, so
// what is left on a stretch of air is the error of nine times as many places
// and not a pattern. @ref `colby_engine::haze`.
@fragment
fn fragment_haze(input: SkyOutput) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(input.clip_position.xy);
    let ray = air_ray(texel * 2);
    let offset = (f32(tile_at(texel, air.rect)) + 0.5) / 9.0;
    let count = min(globals.counts.x, MAX_LAMPS);
    var light = vec3<f32>(0.0);
    var followed = 0u;

    for (var index = 0u; index < count; index++) {
        if (followed >= AIR_LAMPS) {
            break;
        }

        let lamp = globals.lamps[index];
        let span = air_span(lamp, ray);

        if (span.y <= span.x) {
            continue;
        }

        followed += 1u;
        light += air_lamp(lamp, ray, span, offset);
    }

    // **the sun after the lamps, not before them.** A sum that starts at a
    // nought the compiler can see has the first lamp's add folded away; one
    // that starts at the sun's light may have that add fused into the multiply
    // before it, which rounds a hair apart - and it moved a room whose floor
    // hides all of the sun by a level at eight pixels, where the sun sends
    // exactly nothing. Added last, a sun of nought is a fused add of nought
    light += air_sun(ray, offset);

    return vec4<f32>(light, 1.0);
}

// What the air is lit by from everywhere at once: the world's ambient color, or
// its environment averaged over every way there is - the roughest level
// straight up and straight down, which between them cover the whole sphere.
fn air_ambient() -> vec3<f32> {
    if (globals.sky_horizon.w > 0.5) {
        return (ambient_radiance(vec3<f32>(0.0, 1.0, 0.0)) + ambient_radiance(vec3<f32>(0.0, -1.0, 0.0))) * 0.5;
    }

    return globals.ambient.rgb;
}

// What the march found, at the whole picture's size: four texels blended by
// where the pixel lies between them, and a texel that is not in front of the
// pixel's own surface read as whichever of the four is nearest it in distance.
//
// **A pixel at an even place on both axes reads its own texel and nothing
// else**, to the last bit, and the blend is `a + (b - a) * t` so four equal
// texels come back equal.
fn air_blown_up(pixel: vec2<i32>) -> vec3<f32> {
    let first = vec2<i32>(air.rect.xy) / 2;
    let last = (vec2<i32>(air.rect.xy + air.rect.zw) + 1) / 2 - 1;
    let base = pixel / 2;
    let t = vec2<f32>(pixel - base * 2) * 0.5;
    let along = air_distance(textureLoad(air_depth, pixel, 0));
    let places = array<vec2<i32>, 4>(
        clamp(base, first, last),
        clamp(base + vec2<i32>(1, 0), first, last),
        clamp(base + vec2<i32>(0, 1), first, last),
        clamp(base + vec2<i32>(1, 1), first, last),
    );

    var texels = array<vec3<f32>, 4>();
    var own = array<bool, 4>();
    var nearest = 0u;
    var closest = 3.4e38;

    for (var index = 0u; index < 4u; index++) {
        let gap = abs(air_distance(textureLoad(air_depth, places[index] * 2, 0)) - along);

        texels[index] = textureLoad(air_found, places[index], 0).rgb;
        own[index] = gap <= AIR_PLANE * along;

        if (gap < closest) {
            closest = gap;
            nearest = index;
        }
    }

    for (var index = 0u; index < 4u; index++) {
        if (!own[index]) {
            texels[index] = texels[nearest];
        }
    }

    let upper = texels[0] + (texels[1] - texels[0]) * t.x;
    let lower = texels[2] + (texels[3] - texels[2]) * t.x;

    return upper + (lower - upper) * t.y;
}

// The air over the picture: what it sends towards the eye added, and what it
// takes out of the light behind it taken, both in one blend.
//
// **rgb is what the air sends and alpha is what it takes**: the pipeline blends
// `this + picture * (1 - alpha)`, so with alpha the share of the light behind
// that the air scatters away, what comes out is the picture dimmed by the air
// and lit by it. The share is worked out here at every pixel from its own
// distance - `1 - exp(-haze * d)`, exact - and the light is the march's,
// blown back up; the light from everywhere at once is the one part with no
// shadow to cut it, and is worked out in closed form, `ambient * (1 - exp(-haze
// * d))`, which is what the air adds when every way is lit alike.
@fragment
fn fragment_haze_apply(input: SkyOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.clip_position.xy);
    let ray = air_ray(pixel);
    let kept = exp(-air.medium.x * ray.far);

    return vec4<f32>(air_blown_up(pixel) + air_ambient() * (1.0 - kept), 1.0 - kept);
}

// What one point of a surface is made of, before it is lit: what its own
// material says, and then whatever the decals over it painted.
//
// The last two are the material's alone, and no decal moves them: how much of
// the light arriving from everywhere reaches the point, and the light it gives
// off.
struct Surface {
    color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    occlusion: f32,
    emission: vec3<f32>,
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
    // the channels the exchange format puts them in: metal in blue and
    // roughness in green, occlusion in red
    let numbers = textureSample(finish_map, surface_sampler, input.uv);
    let occluded = textureSample(occlusion_map, surface_sampler, set_of(input, OCCLUSION_UV2)).r;
    let glowing = textureSample(glow_map, surface_sampler, set_of(input, GLOW_UV2)).rgb;

    var surface = Surface(
        input.tint.rgb * sampled.rgb,
        input.surface.x * numbers.b,
        input.surface.y * numbers.g,
        shading_normal(input),
        // the exchange format's own arithmetic, which is one at a strength of
        // nought and the picture at a strength of one
        1.0 + finish.glow.a * (occluded - 1.0),
        finish.glow.rgb * glowing,
    );

    if ((input.flags & UNDECALED) == 0u) {
        surface = painted(surface, input.world_position, normalize(input.normal), across, down);
    }

    return surface;
}

// What a bake kept of the light arriving at the point a fragment shades, from
// everywhere at once: rgb in the unit the ambient color is in, and a one.
//
// Asked only by the entry points that draw a thing a bake gave a place, in a
// frame that reads the lightmap: the scene draws everything else with the ones
// that hand in a nought instead. @ref `fragment_baked`.
//
// The level is named, so the read asks nothing of the derivatives and is
// allowed where only some pixels of a quad take it.
fn baked_at(input: VertexOutput) -> vec4<f32> {
    return vec4<f32>(textureSampleLevel(lightmap, lightmap_sampler, input.lightmap, 0.0).rgb, 1.0);
}

// The coordinates a picture the material may lay out on either set is read at.
//
// A select rather than a branch, so that every pixel of a quad samples with
// the same derivatives whichever set it is.
//
// @param bit - which of the two in `finish.flags` says the second set
fn set_of(input: VertexOutput, bit: u32) -> vec2<f32> {
    return select(input.uv, input.uv2, (finish.flags.x & bit) != 0u);
}

// Everything the three entry points that draw a thing with no baked light do
// once the albedo has been sampled.
//
// Returns the color alone. What goes in the alpha channel is the one thing the
// three disagree about, so it is theirs rather than this function's.
//
// **The baked light is a constant nought here, and that is the point**: the
// branch in `lit_at` that reads it folds away when the shader is compiled.
// @ref `fragment_baked` for why that matters.
//
// @param lit - how much of the sky the point sees, which the entry point asks
// for because the masked one has to ask before its discard: @ref `seen`
// @param found - what the point's reflection found on the picture, which the
// entry point asks for because glass asks for none: @ref `found_at`
fn shade(input: VertexOutput, sampled: vec4<f32>, lit: f32, found: vec4<f32>) -> vec3<f32> {
    return shaded(input, sampled, lit, found, vec4<f32>(0.0));
}

// The same for a thing a bake gave a place, which reads what the bake kept.
fn shade_baked(input: VertexOutput, sampled: vec4<f32>, lit: f32, found: vec4<f32>) -> vec3<f32> {
    return shaded(input, sampled, lit, found, baked_at(input));
}

// What both of those do.
//
// @param baked - what a bake kept of the light arriving at the point, a one
// where it kept any: @ref `lit_at`
fn shaded(
    input: VertexOutput,
    sampled: vec4<f32>,
    lit: f32,
    found: vec4<f32>,
    baked: vec4<f32>,
) -> vec3<f32> {
    let surface = surface_at(input, sampled);

    // a surface drawn as its own color: no light on it and none given off, and
    // the fog over it as over everything else. The same over the whole draw,
    // so the branch costs a picture nothing.
    if ((finish.flags.x & UNLIT) != 0u) {
        return fogged(surface.color, input.world_position);
    }

    let towards_eye = normalize(globals.eye.xyz - input.world_position);
    let view_depth = dot(input.world_position - globals.eye.xyz, globals.forward.xyz);
    let slice = cascade_of(view_depth);
    // what the surface gives off, after everything that lights it and before
    // the fog: exposed, bloomed and curved like any other light. Added last,
    // so that a surface giving off nought is the lit one to the bit.
    let color = min(
        lit_at(surface, input.world_position, towards_eye, slice, lit, found, baked)
            + surface.emission,
        vec3<f32>(HDR_CEILING),
    );

    if (globals.shadow.w > 0.5) {
        return color * cascade_color(slice);
    }

    return fogged(color, input.world_position);
}

// How much light one point of a surface sends one way: every light this
// renderer has, from the sun through the lamps to what arrives from everywhere.
//
// **A function of its own because two passes ask it**, `surface_at`'s reason
// from the other end. The scene lights the surface a fragment covers, towards
// the eye; the pass that follows reflections lights the surface a reflected ray
// found, towards the point it was reflected from, out of what the pass before
// the scene wrote down about it. One arithmetic for both is what makes a thing
// seen in a mirror the same thing seen straight on. @ref
// `fragment_reflections`.
//
// Nothing here knows which way the eye is except through `towards_eye`, and
// nothing here fogs or tints: both are the picture's business.
//
// @param surface - what the point is made of, decals painted
// @param world_position - where it is
// @param towards_eye - the way the light is sent, as a unit vector
// @param slice - the cascade the point falls in, @ref `cascade_of`
// @param lit - how much of the sky the point sees
// @param found - what the point's reflection found on the picture, rgb light
// already multiplied by a and a how much of the reflection it stands for:
// nought for a point nothing was followed from, @ref `found_at`
// @param baked - what a bake kept of the light arriving at the point from
// everywhere at once, rgb in the ambient color's unit and a one where it kept
// any: nought for a point with no place on the lightmap, @ref `baked_at`
fn lit_at(
    surface: Surface,
    world_position: vec3<f32>,
    towards_eye: vec3<f32>,
    slice: i32,
    lit: f32,
    found: vec4<f32>,
    baked: vec4<f32>,
) -> vec3<f32> {
    let base_color = surface.color;

    let metallic = clamp(surface.metallic, 0.0, 1.0);
    // held here rather than where the material is read, so that what a person
    // typed is what gets saved, and after the decals, which move it per pixel.
    // @ref `MIN_ROUGHNESS` for how smooth that lets a surface be.
    let roughness = clamp(surface.roughness, MIN_ROUGHNESS, 1.0);

    let normal = surface.normal;
    let towards_light = normalize(-globals.light.xyz);

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
    let reaching = shadowing(world_position, normal, 1.0 - normal_dot_light, slice);

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
            world_position,
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
    // what the environment sends back along the reflection - the sky's level
    // for this roughness, or the one ambient color - which is what a reflection
    // found on the picture takes the place of
    var stand_in: vec3<f32>;

    if (globals.sky_horizon.w > 0.5) {
        let reflected = reflect(-towards_eye, normal);

        indirect = ambient_radiance(normal) * indirect_diffuse
            + reflected_radiance(reflected, roughness) * ambient_specular;
        // read again rather than named once and used in both places: named,
        // the sum above was compiled another way, and a world with an
        // environment moved by a level at two pixels where nothing was found
        stand_in = reflected_radiance(reflected, roughness);
    } else {
        indirect = globals.ambient.rgb * (indirect_diffuse + ambient_specular);
        stand_in = globals.ambient.rgb;
    }

    // what a bake kept takes the place of what the environment sends the
    // diffuse half, and of nothing else. The reflection keeps what the
    // environment sends along it, which is `stand_in` on both sides of the
    // branch above, and everything below reads the one sum as it did: the
    // share of the sky multiplies a baked light as it multiplied the sky it
    // stands in for - it is a radiance, not a third share - and what a
    // reflection found is mixed over the reflection's half as before.
    //
    // **Written over the sum above rather than folded into it.** Where the
    // lightmap holds exactly the ambient color, `L * d + a * s` is not `a * (d
    // + s)` to the last bit, and a thing with no place has to draw the bytes it
    // drew before there were lightmaps: so the sum above is left as it was, a
    // baked point's is written again, and a thing with no place is drawn by an
    // entry point that hands in a constant nought, which folds this away.
    // @ref `fragment_baked`.
    if (baked.a > 0.5) {
        indirect = baked.rgb * indirect_diffuse + stand_in * ambient_specular;
    }

    // how much of the sky this point can see, and it multiplies this term and
    // nothing else, for the shadows' reason turned round: a sun or a lamp is
    // taken away by its own shadow, and what stands in for the light arriving
    // from everywhere is taken away by whatever is near enough to be in the
    // way of everywhere. Both halves by the one number, because the buffer
    // holds one: every second factor in the field is a fit to something it
    // does not hold. After the branch rather than inside each side of it, so
    // that both are multiplied by the same thing the same way.
    //
    // **And the material's own occlusion takes the share's place wherever it
    // is smaller, rather than multiplying it.** Both are pictures of the same
    // crease - one baked with the model, one worked out from the frame - and a
    // product would darken the crease twice. At a material with no occlusion
    // picture the smaller of the two is the share itself, to the bit.
    let open = min(lit, surface.occlusion);

    indirect *= open;

    // what the reflection found on the picture, in the place of the share of
    // what the environment sends along it that the found light stands for:
    // `stand_in * lit * (1 - a) + found` where the sum above has `stand_in *
    // lit`, times the same lobe, which is the environment's reflection and the
    // found one mixed over the light itself rather than over what the surface
    // sends back of it.
    //
    // **What was found is not multiplied by how much of the sky the point
    // sees.** That share stands for the light arriving from far off that
    // something near hides; a found reflection is not far off, it is the near
    // thing, met by a ray and lit where it stands, its own share of the sky
    // included. A floor at the foot of a wall sees less of the sky because of
    // the wall, and the wall is what its reflection found: darkening it again
    // would count the wall twice. What is left of the environment's reflection
    // keeps the share, as it had it.
    //
    // **Added last, and as a difference.** The sum above is what the picture was
    // before any of this existed, and a nought added after it leaves it to the
    // bit - where an add a compiler cannot see is nought, put anywhere else in a
    // sum, may have the add before it fused into its multiply and rounded a hair
    // apart.
    let mixed = (found.rgb - stand_in * (found.a * open)) * ambient_specular;

    // @ref `HDR_CEILING`: past it a smooth highlight would not fit the target.
    return min(direct + indirect + mixed, vec3<f32>(HDR_CEILING));
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
