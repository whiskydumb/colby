// The game interface: rounded rectangles, distance-field text and images, all
// through one pipeline.
//
// Every quad carries what it needs to draw itself, so a whole document is one
// draw call per bound texture rather than one per box. `kind` picks which of
// the three it is:
//
//   0  a rounded rectangle, painted in `color`
//   1  a glyph, sampled out of a font's distance field and tinted `color`
//   2  an image, sampled out of a texture and tinted `color`
//
// Both of the shapes are antialiased against the screen-space derivative of
// their own distance rather than against a fixed width, which is what makes one
// baked atlas legible at every font size a stylesheet asks for.
//
// Every quad also carries what is left of it after `overflow: hidden` on any
// box above it. Cutting here rather than with a scissor rectangle keeps a
// document at one draw call per texture instead of one per clipping box as
// well, and it is what makes clipping able to follow a corner radius: the
// distance function a rounded rectangle is already drawn with is the same one
// the cut is measured against.

struct Screen {
	// the layout area, in layout pixels. Positions are in the same units, so a
	// stylesheet written in pixels is the same size on a scaled display.
	viewport: vec2<f32>,
	_padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> screen: Screen;
@group(1) @binding(0) var atlas: texture_2d<f32>;
@group(1) @binding(1) var atlas_sampler: sampler;

struct VertexIn {
	@location(0) position: vec2<f32>,
	// where this corner sits relative to the middle of its own box, in layout
	// pixels. What the rounded-corner distance is measured from.
	@location(1) local: vec2<f32>,
	@location(2) half_size: vec2<f32>,
	@location(3) uv: vec2<f32>,
	@location(4) color: vec4<f32>,
	// the corner radius for a rectangle, and how many layout pixels the whole
	// range of a distance byte covers for a glyph.
	@location(5) radius: f32,
	@location(6) kind: f32,
	// what is left of this quad after clipping: left, top, right, bottom, in
	// layout pixels, and how round that rectangle's corners are.
	@location(7) clip: vec4<f32>,
	@location(8) clip_radius: f32,
}

struct VertexOut {
	@builtin(position) clip: vec4<f32>,
	@location(0) local: vec2<f32>,
	@location(1) half_size: vec2<f32>,
	@location(2) uv: vec2<f32>,
	@location(3) color: vec4<f32>,
	@location(4) radius: f32,
	@location(5) @interpolate(flat) kind: u32,
	// where this pixel is in layout pixels, which is the space the clip
	// rectangle is in. `@builtin(position)` is in physical pixels and would be
	// the wrong one on a scaled display.
	@location(6) at: vec2<f32>,
	@location(7) @interpolate(flat) clip_rect: vec4<f32>,
	@location(8) @interpolate(flat) clip_radius: f32,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
	var out: VertexOut;

	// pixels with the origin at the top left, into clip space with the origin
	// in the middle and y pointing up.
	let normalized = in.position / max(screen.viewport, vec2<f32>(1.0, 1.0));
	out.clip = vec4<f32>(normalized.x * 2.0 - 1.0, 1.0 - normalized.y * 2.0, 0.0, 1.0);

	out.local = in.local;
	out.half_size = in.half_size;
	out.uv = in.uv;
	out.color = in.color;
	out.radius = in.radius;
	out.kind = u32(in.kind + 0.5);
	out.at = in.position;
	out.clip_rect = in.clip;
	out.clip_radius = in.clip_radius;

	return out;
}

// How far outside a rounded rectangle a point is, in pixels. Negative inside.
fn rounded_distance(local: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
	let corner = max(half_size - vec2<f32>(radius, radius), vec2<f32>(0.0, 0.0));
	let outside = abs(local) - corner;

	return length(max(outside, vec2<f32>(0.0, 0.0)))
		+ min(max(outside.x, outside.y), 0.0)
		- radius;
}

// One pixel's worth of the value, for antialiasing. Guarded because fwidth is
// zero on a degenerate quad, and dividing by it would put NaN on screen.
fn edge_width(value: f32) -> f32 { return max(fwidth(value), 0.0001); }

// How much of this pixel survives the clip rectangle, from zero to one. The
// same rounded distance the shapes use, measured from the middle of the
// rectangle that is doing the cutting rather than from the middle of the quad
// being cut.
fn clip_coverage(at: vec2<f32>, rect: vec4<f32>, radius: f32) -> f32 {
	let half_size = max((rect.zw - rect.xy) * 0.5, vec2<f32>(0.0, 0.0));
	let middle = (rect.xy + rect.zw) * 0.5;
	let corner = min(radius, min(half_size.x, half_size.y));
	let distance = rounded_distance(at - middle, half_size, corner);

	return clamp(0.5 - distance / edge_width(distance), 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
	// sampled unconditionally: a texture read has to happen in uniform control
	// flow, and the result is simply ignored by the kind that does not want it.
	let sample = textureSample(atlas, atlas_sampler, in.uv);

	var alpha = in.color.a;
	var tint = in.color.rgb;

	if in.kind == 1u {
		// the atlas stores a signed distance: 0.5 is the outline, and the whole
		// range covers `radius` layout pixels at the size this is drawn at.
		let distance = (sample.r - 0.5) * in.radius;
		alpha = alpha * clamp(distance / edge_width(distance) + 0.5, 0.0, 1.0);
	} else {
		if in.kind == 2u {
			tint = tint * sample.rgb;
			alpha = alpha * sample.a;
		}

		let distance = rounded_distance(in.local, in.half_size, in.radius);
		alpha = alpha * clamp(0.5 - distance / edge_width(distance), 0.0, 1.0);
	}

	alpha = alpha * clip_coverage(in.at, in.clip_rect, in.clip_radius);

	if alpha <= 0.0 {
		discard;
	}

	return vec4<f32>(tint, alpha);
}
