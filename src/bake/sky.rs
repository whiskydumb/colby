//! What a ray that leaves the world brings back.
//!
//! **The same sky a frame lights with, and chosen by the same rule.** A world
//! whose sky names a cube of six faces is lit by that cube, and one that does
//! not is lit by its one ambient color; a name that does not answer, or answers
//! with a flat picture, is the ambient color too, which is what the renderer
//! falls back to for the same three cases.
//!
//! **One level of the cube, not the sharpest.** The cube's chain is a roughness
//! chain: level nought is the picture, and each level below it is the same sky
//! seen through a rougher lobe. A gather averages what a hundred directions
//! bring back under a cosine, which is a far wider lobe than any of them - so
//! reading a lightly blurred level changes the average by next to nothing and
//! takes away most of what a small bright sun in the picture would otherwise
//! scatter as noise across neighboring points. The level read is the first one
//! no more than [`SIDE`] texels a face.

use colby_core::{
	abi::{
		World,
		texture::{CUBE_FACES, Texel, TextureData},
	},
	glam::Vec3,
};

/// The most texels a side of the face a gather reads may have.
///
/// Sixteen, which for the compiler's cube of 128 is the fourth level, at a
/// roughness of three sevenths.
pub const SIDE: u32 = 16;

/// Which way one face of the cube looks and how its texels are laid out on it:
/// the direction out of its middle, the axis a column walks along and the one a
/// row walks along.
///
/// The asset compiler's own table, in the order the faces are stored -
/// `+x -x +y -y +z -z` - so that the texel a direction lands on here is the one
/// the compiler filled from that direction.
const FACES: [(Vec3, Vec3, Vec3); 6] = [
	(Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y),
	(Vec3::NEG_X, Vec3::Z, Vec3::NEG_Y),
	(Vec3::Y, Vec3::X, Vec3::Z),
	(Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
	(Vec3::Z, Vec3::X, Vec3::NEG_Y),
	(Vec3::NEG_Z, Vec3::NEG_X, Vec3::NEG_Y),
];

/// What lies beyond everything, as a gather reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum Sky {
	/// One color from every direction: the world's ambient color.
	Flat(Vec3),

	/// A cube of six faces, one level of it, decoded.
	Cube {
		/// How many texels a face is on a side.
		side: u32,

		/// Every texel of every face, face after face, row after row, in the
		/// order the file stores them.
		texels: Vec<Vec3>,
	},
}

impl Sky {
	/// The sky a world is lit by.
	///
	/// @param world - whose sky and texture registry are read
	#[must_use]
	pub fn of(world: &World) -> Self {
		if !world.sky.lights() {
			return Self::Flat(world.ambient);
		}

		world
			.textures
			.get(world.sky.cubemap)
			.and_then(|entry| Self::cube(entry.value()))
			.unwrap_or(Self::Flat(world.ambient))
	}

	/// One level of a cube, decoded, if the texture is one.
	///
	/// @param data - what the registry holds
	/// @return the sky, or `None` for anything that is not a whole cube of
	/// half-precision texels
	#[must_use]
	pub fn cube(data: &TextureData) -> Option<Self> {
		if !data.is_cube() || !data.is_consistent() || data.texel != Texel::Rgba16Float {
			return None;
		}

		let count = u32::try_from(data.levels.len()).ok()?;
		let level = (0..count)
			.find(|&level| data.level_size(level).0 <= SIDE)
			.unwrap_or_else(|| count.saturating_sub(1));
		let (side, _) = data.level_size(level);
		let bytes = data.levels.get(usize::try_from(level).ok()?)?;
		let texels = bytes
			.chunks_exact(Texel::Rgba16Float.bytes())
			.map(|texel| {
				let channel =
					|at: usize| widened(u16::from_le_bytes([texel[at * 2], texel[at * 2 + 1]]));

				Vec3::new(channel(0), channel(1), channel(2))
			})
			.collect::<Vec<_>>();
		let expected = usize::try_from(side)
			.ok()?
			.pow(2)
			.checked_mul(usize::try_from(CUBE_FACES).ok()?)?;

		(texels.len() == expected).then_some(Self::Cube { side, texels })
	}

	/// What arrives from one direction.
	///
	/// The face the direction leaves through is the axis it is longest along,
	/// and the place on that face is where it crosses it; the four nearest
	/// texels are blended, held at the face's own edge rather than reaching
	/// round the corner into the next face - which moves the answer by a part
	/// of one texel's worth at the seams and not at all anywhere else.
	///
	/// @param way - the direction, of any length but nought
	#[must_use]
	pub fn toward(&self, way: Vec3) -> Vec3 {
		let (side, texels) = match self {
			| Self::Flat(color) => return *color,
			| Self::Cube { side, texels } => (*side, texels),
		};

		let (face, major) = face_of(way);
		let Some(&(_, right, down)) = FACES.get(face) else {
			return Vec3::ZERO;
		};

		// from minus one to one across the face, then texels
		let span = f32::from(u16::try_from(side).unwrap_or(1)).max(1.0);
		let across = way.dot(right) / major;
		let deep = way.dot(down) / major;
		let x = (across + 1.0) * 0.5 * span;
		let y = (deep + 1.0) * 0.5 * span;
		// the texel whose middle is at a place is the one it falls in, less a half
		let (x, y) = (x - 0.5, y - 0.5);
		let (left, top) = (x.floor(), y.floor());
		let (towards_x, towards_y) = (x - left, y - top);
		let at = |column: f32, row: f32| {
			let column = held(column, side);
			let row = held(row, side);
			let index = face * side_of(side).pow(2) + row * side_of(side) + column;

			texels.get(index).copied().unwrap_or(Vec3::ZERO)
		};

		let upper = at(left, top).lerp(at(left + 1.0, top), towards_x);
		let lower = at(left, top + 1.0).lerp(at(left + 1.0, top + 1.0), towards_x);

		upper.lerp(lower, towards_y)
	}
}

/// Which face a direction leaves through, and how far along its longest axis
/// it goes: the first of two equal axes wins, x before y before z.
fn face_of(way: Vec3) -> (usize, f32) {
	let size = way.abs();

	if size.x >= size.y && size.x >= size.z {
		(usize::from(way.x < 0.0), size.x)
	} else if size.y >= size.z {
		(2 + usize::from(way.y < 0.0), size.y)
	} else {
		(4 + usize::from(way.z < 0.0), size.z)
	}
}

/// A side as an index count.
fn side_of(side: u32) -> usize { usize::try_from(side).unwrap_or(0) }

/// A column or a row, held inside the face.
#[expect(
	clippy::as_conversions,
	clippy::cast_possible_truncation,
	clippy::cast_sign_loss,
	reason = "clamped into the face first, so the value is a whole number inside it"
)]
fn held(value: f32, side: u32) -> usize {
	let last = f64::from(side.saturating_sub(1));

	(f64::from(value).clamp(0.0, last) as usize).min(side_of(side).saturating_sub(1))
}

/// One of the format's sixteen-bit values as a float, exactly.
///
/// Assembled from the bits rather than worked out with a power, so that the
/// answer is the format's own value on every machine: a normal value is the
/// same sign, the exponent moved to a float's bias and the mantissa moved to a
/// float's width, and a value too small to be normal is its mantissa times two
/// to the minus twenty-four, a product by a power of two and therefore exact.
#[must_use]
pub fn widened(bits: u16) -> f32 {
	let sign = u32::from(bits & 0x8000) << 16;
	let exponent = u32::from((bits >> 10) & 0x1F);
	let mantissa = u32::from(bits & 0x03FF);

	match exponent {
		| 0 => {
			let tiny =
				f32::from(u16::try_from(mantissa).unwrap_or(0)) * f32::from_bits(0x3380_0000);

			if sign == 0 { tiny } else { -tiny }
		},
		// an infinity or not a number, which the format keeps as the float's
		| 0x1F => f32::from_bits(sign | 0x7F80_0000 | (mantissa << 13)),
		| _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
	}
}

#[cfg(test)]
mod tests {
	use colby_core::utils::half::half;

	use super::*;

	/// A cube whose every face is one color, the colors in face order, one
	/// level of the given side.
	fn painted(side: u32, colors: [Vec3; 6]) -> TextureData {
		let count = usize::try_from(side * side).expect("a small face");
		let bytes = colors
			.iter()
			.flat_map(|color| texel(*color).repeat(count))
			.collect();

		TextureData {
			width: side,
			height: side,
			faces: CUBE_FACES,
			texel: Texel::Rgba16Float,
			levels: vec![bytes],
		}
	}

	/// One opaque texel of a color, as the format stores it.
	fn texel(color: Vec3) -> Vec<u8> {
		[color.x, color.y, color.z, 1.0]
			.iter()
			.flat_map(|channel| half(*channel).to_le_bytes())
			.collect()
	}

	/// The asset compiler's table of faces, written out again from its own
	/// file rather than taken from the one under test: the direction out of
	/// each face's middle, the way a column walks and the way a row walks.
	/// A test built on the table it checks would agree with any mistake in it.
	const COMPILED: [[[f32; 3]; 3]; 6] = [
		[[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
		[[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
		[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
		[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
		[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
		[[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
	];

	/// Six colors a face can be told apart by.
	const COLORS: [Vec3; 6] = [
		Vec3::new(1.0, 0.0, 0.0),
		Vec3::new(0.0, 1.0, 0.0),
		Vec3::new(0.0, 0.0, 1.0),
		Vec3::new(1.0, 1.0, 0.0),
		Vec3::new(0.0, 1.0, 1.0),
		Vec3::new(1.0, 0.0, 1.0),
	];

	#[test]
	fn a_direction_reads_the_face_it_leaves_through() {
		let sky = Sky::cube(&painted(4, COLORS)).expect("a cube");
		let ways = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];

		for (way, color) in ways.into_iter().zip(COLORS) {
			assert_eq!(sky.toward(way), color, "straight along {way}");
			assert_eq!(
				sky.toward(way * 3.0 + Vec3::splat(0.1)),
				color,
				"and a little off it, at any length"
			);
		}
	}

	#[test]
	fn a_texel_is_read_where_the_compiler_wrote_it() {
		// every texel a color of its own, and the direction through the middle
		// of each worked out the way the asset compiler works it out
		let side = 4_u32;
		// face, row and column of every texel in the order they are stored,
		// and a color that is those three numbers
		let expected: Vec<(u32, u32, u32, Vec3)> = (0..6 * side * side)
			.map(|index| {
				let (face, row, column) =
					(index / (side * side), index / side % side, index % side);
				let number = |value: u32| f32::from(u16::try_from(value).expect("small"));

				(face, row, column, Vec3::new(number(face), number(row), number(column)))
			})
			.collect();
		let bytes = expected
			.iter()
			.flat_map(|(.., color)| texel(*color))
			.collect();

		let sky = Sky::cube(&TextureData {
			width: side,
			height: side,
			faces: CUBE_FACES,
			texel: Texel::Rgba16Float,
			levels: vec![bytes],
		})
		.expect("a cube");

		for (face, row, column, color) in expected {
			let [forward, right, down] =
				COMPILED[usize::try_from(face).expect("six")].map(Vec3::from_array);
			let middle = |index: u32| {
				let fraction = (f32::from(u16::try_from(index).expect("four")) + 0.5) / 4.0;

				fraction + fraction - 1.0
			};
			let way = forward + right * middle(column) + down * middle(row);

			assert_eq!(sky.toward(way), color, "face {face}, row {row}, column {column}");
		}
	}

	#[test]
	fn a_world_with_no_cube_is_lit_by_its_ambient_color() {
		let mut world = World::new();

		world.ambient = Vec3::new(0.2, 0.3, 0.4);

		assert_eq!(Sky::of(&world), Sky::Flat(world.ambient), "no sky named");
		assert_eq!(
			Sky::of(&world).toward(Vec3::new(0.3, -0.8, 0.1)),
			world.ambient,
			"the same color whichever way"
		);
	}

	#[test]
	fn a_flat_picture_named_as_a_cube_is_not_a_sky() {
		assert!(Sky::cube(&TextureData::white()).is_none(), "one face is no cube");
	}

	#[test]
	fn the_widest_level_read_is_no_wider_than_the_side_allowed() {
		let mut data = painted(64, COLORS);

		for side in [32_u32, 16, 8, 4, 2, 1] {
			data.levels
				.push(painted(side, COLORS).levels.remove(0));
		}

		let Some(Sky::Cube { side, .. }) = Sky::cube(&data) else {
			panic!("a chain of seven levels is a cube");
		};

		assert_eq!(side, SIDE, "the first level no wider than {SIDE}");
	}

	#[test]
	fn a_half_widens_to_the_value_that_narrowed_into_it() {
		for value in [0.0_f32, 1.0, -2.5, 0.333_251_95, 65504.0, 6.103_515_6e-5, 5.960_464_5e-8] {
			assert_eq!(
				widened(half(value)).to_bits(),
				value.to_bits(),
				"{value} survives the round trip"
			);
		}

		assert_eq!(widened(0x3C00).to_bits(), 1.0_f32.to_bits(), "one is 0x3c00");
		assert_eq!(
			widened(0x0001).to_bits(),
			5.960_464_5e-8_f32.to_bits(),
			"and the smallest value is two to the minus 24"
		);
		assert!(widened(0x7C00).is_infinite(), "an infinity stays one");
	}
}
