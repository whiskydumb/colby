// The depth buffer made readable by a pass that is not the one that wrote it.
//
// At one sample a pixel nothing here runs: the buffer the scene tested against
// is a texture like any other, and whatever reads depth after the scene binds
// it. At four there is no such texture. A multisampled one is read a sample at
// a time through a binding of its own type, and nothing resolves a depth
// format on the way out of a pass, so this pass writes the nearest of each
// pixel's samples into a buffer of one sample, and that is what is read.
//
// **The nearest rather than the mean or the first.** A pixel on an edge holds
// samples of two surfaces, and a reader wants the depth of one of them. The
// mean is a depth no surface has, which a blur by distance turns into a halo;
// the first sample sits wherever it sits in the pixel, which makes the edge a
// staircase again. The nearest keeps a silhouette in front of what is behind
// it, which is what a light shaft needs from its mask and what a blur by
// distance needs from a near edge.

@group(0) @binding(0) var multi: texture_depth_multisampled_2d;

// One triangle covering the target, out of nothing but the vertex index.
@vertex
fn vertex_screen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;

    return vec4<f32>(x, y, 0.0, 1.0);
}

// The nearest of a pixel's samples, as the depth of the one that stands for
// them.
//
// Nearest is least: the buffer is cleared to one at the far plane, and a
// nearer surface wins the scene's test by comparing less.
@fragment
fn fragment_nearest(@builtin(position) at: vec4<f32>) -> @builtin(frag_depth) f32 {
    let texel = vec2<i32>(at.xy);
    let count = i32(textureNumSamples(multi));
    var nearest = textureLoad(multi, texel, 0);

    for (var sample = 1; sample < count; sample += 1) {
        nearest = min(nearest, textureLoad(multi, texel, sample));
    }

    return nearest;
}
