//! Turning a shape into a signed distance field.
//!
//! A glyph rasterized at one size is only crisp at that size. A field of
//! distances is crisp at any of them: the shader compares the sampled distance
//! against zero and antialiases against however fast the distance is changing
//! on screen, so one baked atlas serves every `font-size` in a stylesheet. That
//! is the whole reason this module exists, and it is why the font compiler can
//! bake once instead of once per size the interface happens to ask for.
//!
//! The algorithm is the eight-point sequential signed Euclidean distance
//! transform - two sweeps over the image, each carrying an offset to the
//! nearest seed pixel forward from its neighbors. It is linear in the number of
//! pixels, which matters: the alternative, measuring every pixel against every
//! outline segment, is what turns a font bake from a fifth of a second into
//! most of a minute.
//!
//! It is approximate. A sweep can miss the true nearest seed in shapes with
//! deep concavities, by about one pixel in the cases it gets wrong at all -
//! well under the supersampling the font compiler feeds it with, and invisible
//! once the field is clamped to a few pixels either side of the edge.

/// The offset stored for a pixel with no seed anywhere near it.
///
/// Far enough that any real offset in an image this module will ever see beats
/// it, and small enough that its squared length stays inside an `i32`.
const FAR: i16 = 16384;

/// A pixel's offset to the nearest seed, in pixels.
type Offset = [i16; 2];

/// The offset that means "nothing found yet".
const NOWHERE: Offset = [FAR, FAR];

/// The offset a seed pixel holds: itself.
const HERE: Offset = [0, 0];

/// How far each pixel is from the edge of the shape, in pixels.
///
/// Positive inside the shape and negative outside it, which is the sign
/// convention the atlas stores and the shader reads. The magnitude is the
/// distance to the nearest pixel of the other region, so it is quantized to
/// pixel centers - feed this a supersampled image and the quantization divides
/// out with everything else.
///
/// @param inside - one flag per pixel, row major, `true` inside the shape
/// @param width - how many pixels a row holds
/// @param height - how many rows there are
/// @return one distance per pixel, in the same order; empty if the size does
/// not match the flags
#[must_use]
pub fn signed_distance(inside: &[bool], width: usize, height: usize) -> Vec<f32> {
	if width == 0 || height == 0 || inside.len() != width * height {
		return Vec::new();
	}

	let mut to_inside = Grid::seeded(inside, width, height, true);
	let mut to_outside = Grid::seeded(inside, width, height, false);

	to_inside.resolve();
	to_outside.resolve();

	to_outside
		.points
		.iter()
		.zip(to_inside.points.iter())
		.map(|(outside, inside)| length(*outside) - length(*inside))
		.collect()
}

/// One sweep's working state: an offset per pixel.
struct Grid {
	width: usize,
	height: usize,
	points: Vec<Offset>,
}

impl Grid {
	/// A grid with zero offsets on one region and nothing on the other.
	///
	/// @param inside - the shape
	/// @param width - row length
	/// @param height - row count
	/// @param seed - which region the seeds go in: `true` for the inside
	fn seeded(inside: &[bool], width: usize, height: usize, seed: bool) -> Self {
		Self {
			width,
			height,
			points: inside
				.iter()
				.map(|&it| if it == seed { HERE } else { NOWHERE })
				.collect(),
		}
	}

	/// Fills in every pixel's offset to the nearest seed.
	fn resolve(&mut self) {
		self.sweep_down();
		self.sweep_up();
	}

	/// Carries offsets down and to the right, then back along each row.
	fn sweep_down(&mut self) {
		for y in 0..self.height {
			for x in 0..self.width {
				let mut best = self.at(x, y);
				self.compare(&mut best, x, y, -1, 0);
				self.compare(&mut best, x, y, 0, -1);
				self.compare(&mut best, x, y, -1, -1);
				self.compare(&mut best, x, y, 1, -1);
				self.put(x, y, best);
			}

			for x in (0..self.width).rev() {
				let mut best = self.at(x, y);
				self.compare(&mut best, x, y, 1, 0);
				self.put(x, y, best);
			}
		}
	}

	/// The same, upwards, which is what makes the answer symmetric.
	fn sweep_up(&mut self) {
		for y in (0..self.height).rev() {
			for x in (0..self.width).rev() {
				let mut best = self.at(x, y);
				self.compare(&mut best, x, y, 1, 0);
				self.compare(&mut best, x, y, 0, 1);
				self.compare(&mut best, x, y, -1, 1);
				self.compare(&mut best, x, y, 1, 1);
				self.put(x, y, best);
			}

			for x in 0..self.width {
				let mut best = self.at(x, y);
				self.compare(&mut best, x, y, -1, 0);
				self.put(x, y, best);
			}
		}
	}

	/// Takes a neighbor's answer if it is nearer than the one already held.
	///
	/// The neighbor's offset points at a seed from *its* position, so stepping
	/// back to this pixel is what the addition is for.
	fn compare(&self, best: &mut Offset, x: usize, y: usize, step_x: i16, step_y: i16) {
		let Some(mut other) = self.neighbor(x, y, step_x, step_y) else {
			return;
		};

		other[0] = other[0].saturating_add(step_x);
		other[1] = other[1].saturating_add(step_y);

		if length_squared(other) < length_squared(*best) {
			*best = other;
		}
	}

	/// A neighbor's offset, or `None` when the step leaves the image.
	fn neighbor(&self, x: usize, y: usize, step_x: i16, step_y: i16) -> Option<Offset> {
		let x = x.checked_add_signed(isize::from(step_x))?;
		let y = y.checked_add_signed(isize::from(step_y))?;

		if x >= self.width || y >= self.height {
			return None;
		}

		Some(self.at(x, y))
	}

	/// One pixel's offset.
	fn at(&self, x: usize, y: usize) -> Offset {
		self.points
			.get(y * self.width + x)
			.copied()
			.unwrap_or(NOWHERE)
	}

	/// Writes one pixel's offset.
	fn put(&mut self, x: usize, y: usize, point: Offset) {
		if let Some(slot) = self.points.get_mut(y * self.width + x) {
			*slot = point;
		}
	}
}

/// How far an offset reaches, squared, which is what comparisons need.
fn length_squared(point: Offset) -> i32 {
	let x = i32::from(point[0]);
	let y = i32::from(point[1]);

	x * x + y * y
}

/// How far an offset reaches.
fn length(point: Offset) -> f32 { f32::from(point[0]).hypot(f32::from(point[1])) }

#[cfg(test)]
mod tests {
	use super::*;

	/// A filled disk, as flags.
	fn disk(size: usize, radius: f32) -> Vec<bool> {
		let center = f32::from(u16::try_from(size).unwrap_or(0)) / 2.0 - 0.5;

		(0..size * size)
			.map(|index| {
				let x = f32::from(u16::try_from(index % size).unwrap_or(0)) - center;
				let y = f32::from(u16::try_from(index / size).unwrap_or(0)) - center;

				x.hypot(y) <= radius
			})
			.collect()
	}

	#[test]
	fn a_field_of_the_wrong_size_is_refused_rather_than_read_past() {
		assert!(signed_distance(&[true; 4], 3, 3).is_empty(), "four flags is not a 3x3 image");
		assert!(signed_distance(&[], 0, 0).is_empty(), "and nothing is not an image at all");
	}

	#[test]
	fn everything_inside_is_positive_and_everything_outside_is_negative() {
		let size = 32;
		let flags = disk(size, 10.0);
		let field = signed_distance(&flags, size, size);

		for (index, &is_inside) in flags.iter().enumerate() {
			let distance = field.get(index).copied().unwrap_or(0.0);

			assert_eq!(
				is_inside,
				distance > 0.0,
				"pixel {index} is {} the shape and its distance is {distance}",
				if is_inside { "inside" } else { "outside" }
			);
		}
	}

	#[test]
	fn the_distance_at_the_middle_of_a_disk_is_its_radius() {
		let size = 64;
		let radius = 20.0;
		let field = signed_distance(&disk(size, radius), size, size);
		let middle = field
			.get((size / 2) * size + size / 2)
			.copied()
			.unwrap_or(0.0);

		assert!(
			(middle - radius).abs() < 2.0,
			"the middle of a disk of {radius} should be about that far from the edge, got \
			 {middle}"
		);
	}

	#[test]
	fn the_field_falls_off_by_about_one_per_pixel_away_from_the_edge() {
		let size = 64;
		let field = signed_distance(&disk(size, 20.0), size, size);
		let row = size / 2;
		let at = |x: usize| field.get(row * size + x).copied().unwrap_or(0.0);

		// walking out along the middle row from the center: each step should
		// cost about one unit of distance, which is what makes the field usable
		// as a distance rather than as a blurred mask.
		let step = at(size / 2 + 5) - at(size / 2 + 6);

		assert!(
			(step - 1.0).abs() < 0.35,
			"one pixel of travel should be about one unit of distance, got {step}"
		);
	}

	#[test]
	fn a_shape_that_fills_the_image_has_no_outside_to_measure_against() {
		let field = signed_distance(&[true; 16], 4, 4);

		assert!(
			field.iter().all(|&distance| distance > 0.0),
			"every pixel is inside, so every distance is positive rather than zero"
		);
	}
}
