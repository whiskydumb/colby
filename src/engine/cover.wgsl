// What is wholly behind what the pass before the scene drew, and the scene's
// lists drawn without it.
//
// Six entry points in one compute pass, between the pass before the scene and
// the scene's own. The first two fold the depth that pass wrote into a pyramid
// of farthest depths, half the picture on each axis and then half of that until
// one texel is left. The third asks of every thing in the picture's lists
// whether its box lies behind the farthest depth over every pixel it could
// touch. The fourth and fifth count what is kept in every sixteen things and
// before them, and the sixth copies what is kept into a second run of
// placements, batch by batch in the order they came, and says how many of each
// batch are left. @ref `colby_engine::cover`.
//
// **The depth this frame drew, not the last.** Everything in the picture's own
// lists was drawn into that depth a moment ago, so a thing that has just come
// out from behind a wall is in it already and its box is in front of what the
// wall left - nothing here is a frame late, and nothing needs a history.
//
// **Farther is larger.** The depth is stored the way the scene's is, nought at
// the near plane and one at the far one, so the depth that stands for a region
// of pixels is the largest in it, and a box is behind it only when the nearest
// of its corners is larger still.

struct Tuning {
    // World space into clip space: the matrix the picture is drawn through.
    view_projection: mat4x4<f32>,
    // The rectangle of the target the picture is drawn into, as x, y, width
    // and height in pixels.
    rect: vec4<f32>,
    // x is how many things the picture's lists hold, y how many levels the
    // pyramid has over the rectangle.
    counts: vec4<u32>,
};

// Which level a pass of the pyramid writes, nought the first.
struct Level {
    index: u32,
    spare_a: u32,
    spare_b: u32,
    spare_c: u32,
};

// One thing in the picture's lists: its box as it stands in the world, the
// middle and the three half-edges, and where it is drawn.
struct Reach {
    center: vec3<f32>,
    // Which of the frame's batches draws it: the solid ones first, then the
    // blended ones, in the order the scene draws them.
    batch: u32,
    x: vec3<f32>,
    // Where that batch's run of placements begins.
    first: u32,
    y: vec3<f32>,
    // How many triangles its mesh has, for the count.
    triangles: u32,
    z: vec3<f32>,
    spare: u32,
};

// How much the pass left out, added to by every invocation at once.
struct Tally {
    instances: atomic<u32>,
    triangles: atomic<u32>,
};

// One group, and each pipeline's layout names the bindings its entry point
// reads: the pyramid's two passes 0 to 4, the test, the count and the copy 0 and
// 5 to 12.
@group(0) @binding(0) var<uniform> tuning: Tuning;
@group(0) @binding(1) var<uniform> level: Level;

// The depth the pass before the scene wrote, one sample a pixel, which the
// first level is folded from.
@group(0) @binding(2) var depth: texture_depth_2d;

// The level below the one being written, which every level after the first is
// folded from.
@group(0) @binding(3) var below: texture_2d<f32>;

// The level being written.
@group(0) @binding(4) var written: texture_storage_2d<r32float, write>;

// Every level at once, which the test reads.
@group(0) @binding(5) var pyramid: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> reaches: array<Reach>;
@group(0) @binding(7) var<storage, read_write> kept: array<u32>;

// The scene's placements, a word at a time, and the run the kept ones are copied
// into.
@group(0) @binding(8) var<storage, read> placements: array<u32>;
@group(0) @binding(9) var<storage, read_write> compacted: array<u32>;

// One draw command a batch, five words each, of which this writes the second.
@group(0) @binding(10) var<storage, read_write> commands: array<u32>;
@group(0) @binding(11) var<storage, read_write> tally: Tally;

// How many kept things each chunk of the lists holds, the first `CHUNKS` words,
// and how many come before it, the next `CHUNKS`.
@group(0) @binding(12) var<storage, read_write> chunks: array<u32>;

// How many words one placement is: the `Placement` the scene writes, 128 bytes.
const PLACEMENT_WORDS: u32 = 32u;

// How many words one draw command is, and which of them is the instance count.
const COMMAND_WORDS: u32 = 5u;
const INSTANCE_COUNT: u32 = 1u;

// How many things of the lists one chunk of the count is, and how many chunks
// the most things a world holds make.
//
// A thing's place in its batch's run is how many kept things come before it in
// the batch, and a batch can be most of the lists - a street of houses full of
// one kind of ball is one batch of six hundred. **Every invocation runs at once,
// so what a dispatch costs is its longest loop**: counted thing by thing back to
// the batch's start that loop is as long as the batch, and measured it was
// nearly half the pass; counted back to the start of the lists in chunks it was
// as long as the lists, and dearer still. So no loop here is longer than a
// chunk, or than the chunks: sixteen things a chunk counted where they lie, the
// chunks before each added up, and fifteen things at most counted twice by the
// copy.
const CHUNK: u32 = 16u;
const CHUNKS: u32 = 64u;

// How much farther than the farthest depth a box's nearest corner has to be
// before it counts as behind it.
//
// The corners are worked out here and the depth by the rasterizer, each in
// single precision along its own path, so a box whose face is the face of the
// thing it stands for can come out a few units in the last place either side of
// it. A millionth is a few dozen of those near the far plane; at fifty units
// from an eye with a near plane a tenth away it is two and a half centimeters of
// world, so nothing closer than that behind a wall is left out.
const DEPTH_SLACK: f32 = 1.0e-6;

// How many texels a level of the pyramid has across the rectangle and down it.
//
// Every level halves the one below it, rounding up, so that a texel's two by
// two below it always exists; the texture the levels live in is bigger than
// this, and nothing past it is read.
fn level_size(index: u32, rect: vec2<u32>) -> vec2<u32> {
    let shift = index + 1u;
    let up = (1u << shift) - 1u;

    return max((rect + vec2<u32>(up)) >> vec2<u32>(shift), vec2<u32>(1u));
}

// The first level: the farthest of each two by two of the rectangle's pixels.
//
// A last row or column that has no neighbor past the rectangle's edge takes its
// own depth again, which changes no farthest.
@compute @workgroup_size(8, 8)
fn pyramid_first(@builtin(global_invocation_id) id: vec3<u32>) {
    let rect = vec2<u32>(tuning.rect.zw);
    let size = level_size(0u, rect);

    if any(id.xy >= size) {
        return;
    }

    let origin = vec2<i32>(tuning.rect.xy);
    let last = vec2<i32>(rect) - 1;
    let at = vec2<i32>(id.xy) * 2;
    var farthest = 0.0;

    for (var down = 0; down < 2; down += 1) {
        for (var across = 0; across < 2; across += 1) {
            let pixel = origin + min(at + vec2<i32>(across, down), last);

            farthest = max(farthest, textureLoad(depth, pixel, 0));
        }
    }

    textureStore(written, vec2<i32>(id.xy), vec4<f32>(farthest, 0.0, 0.0, 1.0));
}

// Every level after it: the farthest of each two by two of the level below.
@compute @workgroup_size(8, 8)
fn pyramid_next(@builtin(global_invocation_id) id: vec3<u32>) {
    let rect = vec2<u32>(tuning.rect.zw);
    let size = level_size(level.index, rect);

    if any(id.xy >= size) {
        return;
    }

    let last = vec2<i32>(level_size(level.index - 1u, rect)) - 1;
    let at = vec2<i32>(id.xy) * 2;
    var farthest = 0.0;

    for (var down = 0; down < 2; down += 1) {
        for (var across = 0; across < 2; across += 1) {
            let texel = min(at + vec2<i32>(across, down), last);

            farthest = max(farthest, textureLoad(below, texel, 0).r);
        }
    }

    textureStore(written, vec2<i32>(id.xy), vec4<f32>(farthest, 0.0, 0.0, 1.0));
}

// Whether all of a box lies behind what the pass before the scene drew.
//
// **Asked of the box's eight corners**: the nearest of them is as near as any
// point of the box comes, because a depth grows with distance along the view
// and the box is convex, and the rectangle around where they land holds every
// pixel the box can cover. That rectangle is grown by a pixel on every side,
// because the scene may take four samples a pixel where the pass before it took
// one in the middle: a surface leaning away has a depth at a sample off the
// middle that is farther than at the middle, and the next pixel's middle is
// farther still.
//
// **Read at one level, over at most five texels a side**: the level where the
// rectangle's longer side is four texels or fewer. A box any of whose corners
// is at or behind the eye, or in front of the near plane, is one the near plane
// cuts, and it is never left out.
fn behind(reach: Reach) -> bool {
    var low = vec2<f32>(3.0e38);
    var high = vec2<f32>(-3.0e38);
    var nearest = 1.0;

    for (var corner = 0u; corner < 8u; corner += 1u) {
        let way = vec3<f32>(
            select(-1.0, 1.0, (corner & 1u) != 0u),
            select(-1.0, 1.0, (corner & 2u) != 0u),
            select(-1.0, 1.0, (corner & 4u) != 0u),
        );
        let at = reach.center + reach.x * way.x + reach.y * way.y + reach.z * way.z;
        let clip = tuning.view_projection * vec4<f32>(at, 1.0);

        // written so that a corner that is not a number is a box that is drawn
        //
        // @note: either half alone answers the same through every lens a
        // camera builds, where a z at or past nought is a w at or past the near
        // plane and a corner nearer than that has a depth below every farthest;
        // a mutation pass that took out either passed everything.
        if !(clip.w > 0.0) || !(clip.z >= 0.0) {
            return false;
        }

        let ndc = clip.xyz / clip.w;
        let pixel = tuning.rect.xy + (ndc.xy * vec2<f32>(0.5, -0.5) + 0.5) * tuning.rect.zw;

        nearest = min(nearest, ndc.z);
        low = min(low, pixel);
        high = max(high, pixel);
    }

    let span = tuning.rect.zw;
    let lowest = low - tuning.rect.xy;
    let highest = high - tuning.rect.xy;

    // wholly off the rectangle is the frustum's to leave out, and a box that
    // lands there is drawn rather than tested against a texel it does not reach
    //
    // @note: only a box the frustum's slack let through lands there, and a
    // sliver that thin covers no sample; a mutation pass that took the line out
    // passed everything.
    if !all(highest >= vec2<f32>(0.0)) || !all(lowest < span) {
        return false;
    }

    let first = vec2<u32>(clamp(floor(lowest) - 1.0, vec2<f32>(0.0), span - 1.0));
    let last = vec2<u32>(clamp(floor(highest) + 1.0, vec2<f32>(0.0), span - 1.0));
    let side = max(last.x - first.x, last.y - first.y) + 1u;

    // the level whose texels, 2^(level + 1) pixels across, fit the longer side
    // four times or fewer: one past a power of two needs the next level up
    let fits = 32u - countLeadingZeros(side - 1u);
    let index = min(max(fits, 3u) - 3u, tuning.counts.y - 1u);
    let size = level_size(index, vec2<u32>(span));
    let start = first >> vec2<u32>(index + 1u);
    let end = min(last >> vec2<u32>(index + 1u), size - 1u);
    var farthest = 0.0;

    for (var down = start.y; down <= end.y; down += 1u) {
        for (var across = start.x; across <= end.x; across += 1u) {
            let texel = vec2<i32>(vec2<u32>(across, down));

            farthest = max(farthest, textureLoad(pyramid, texel, i32(index)).r);
        }
    }

    return nearest > farthest + DEPTH_SLACK;
}

// Whether each thing is kept, and how much was left out.
@compute @workgroup_size(64)
fn test(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= tuning.counts.x {
        return;
    }

    let reach = reaches[index];
    let hidden = behind(reach);

    kept[index] = select(1u, 0u, hidden);

    if hidden {
        atomicAdd(&tally.instances, 1u);
        atomicAdd(&tally.triangles, reach.triangles);
    }
}

// How many kept things each chunk of the lists holds.
@compute @workgroup_size(64)
fn count(@builtin(global_invocation_id) id: vec3<u32>) {
    let chunk = id.x;
    let things = tuning.counts.x;
    let start = chunk * CHUNK;

    if start >= things {
        return;
    }

    var counted = 0u;

    // @note: the end is held at the lists' only for the last chunk, whose count
    // nothing adds up, since a chunk is given only what comes before it; a
    // mutation pass that took the hold out passed everything.
    for (var index = start; index < min(start + CHUNK, things); index += 1u) {
        counted += kept[index];
    }

    chunks[chunk] = counted;
}

// How many kept things come before each chunk, from what the count wrote.
@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) id: vec3<u32>) {
    let chunk = id.x;

    if chunk * CHUNK >= tuning.counts.x {
        return;
    }

    var counted = 0u;

    for (var earlier = 0u; earlier < chunk; earlier += 1u) {
        counted += chunks[earlier];
    }

    chunks[CHUNKS + chunk] = counted;
}

// How many kept things come before a place in the lists.
fn kept_before(at: u32) -> u32 {
    let chunk = at / CHUNK;
    var counted = chunks[CHUNKS + chunk];

    for (var index = chunk * CHUNK; index < at; index += 1u) {
        counted += kept[index];
    }

    return counted;
}

// What is kept copied into its batch's run, in the order it was sorted in, and
// the last of a batch saying how many of it there are.
//
// **The order is what keeps the picture the picture.** Two things of one batch
// at exactly one depth are settled by which was drawn first, and a thing's place
// in its batch's run here is how many kept things came before it there.
@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    let count = tuning.counts.x;

    if index >= count {
        return;
    }

    let reach = reaches[index];
    let place = kept_before(index) - kept_before(reach.first);
    let here = kept[index];

    if here != 0u {
        let source = index * PLACEMENT_WORDS;
        let into = (reach.first + place) * PLACEMENT_WORDS;

        for (var word = 0u; word < PLACEMENT_WORDS; word += 1u) {
            compacted[into + word] = placements[source + word];
        }
    }

    let closes = index + 1u == count || reaches[min(index + 1u, count - 1u)].batch != reach.batch;

    if closes {
        commands[reach.batch * COMMAND_WORDS + INSTANCE_COUNT] = place + here;
    }
}
