//! Where a thing that walks may stand, and how it gets from here to there.
//!
//! A **grid of standable cells** rather than a mesh of convex polygons, and
//! that is the one decision everything else here follows from. What a
//! polygonal navmesh buys is a small description of an arbitrary level - the
//! rasterizing, region growing and contour tracing that Recast spends eight
//! thousand lines on all exist to turn a triangle soup into few large
//! polygons. colby's walkable world is a [`Terrain`](super::terrain::Terrain),
//! which is *already* a regular grid, so building a second irregular
//! description over the first buys nothing and costs the whole pipeline.
//! Wicked's `wiPathQuery` is the precedent, and it is nine hundred lines
//! against Unreal's forty-four thousand.
//!
//! **The bake reads the collision world, not the drawn one.** Three of the
//! four engines that bake a navmesh at all take their geometry from
//! collision: s&box hands its baker a whole `PhysicsWorld`, Unreal reads
//! `UBodySetup` through `UNavCollisionBase`, and Godot's default parses both.
//! The reason is the one that matters here: what a character may stand on is
//! what a character stops against, and a navmesh built from anything else is a
//! promise the solver never made. Here that is one downward trace per column,
//! which is the `Collider` grid from step 5l doing what its own comment said
//! it would.
//!
//! **One surface a column, and that is the limit worth knowing.** A bake takes
//! the topmost thing a downward ray meets, so a crate is somewhere to walk and
//! the floor underneath it is not, and a room with a roof on it is invisible
//! below the roof. Recast answers this by keeping every span a column has and
//! Wicked by voxelizing the whole volume; both are a second dimension in the
//! storage and in the query, and neither is what a landscape with things
//! standing on it needs. It is also why there is no headroom setting: with one
//! surface a column there is by construction nothing above the surface to duck
//! under, and a `nav.height` would have been a number that could never have
//! changed an answer.
//!
//! **There is no region to mark.** Unreal needs an `ANavMeshBoundsVolume`
//! because its level is kilometers and baking all of it is not a thing anybody
//! can afford; colby's world is however big its static bodies are, so that is
//! what gets baked and there is nothing for a person to draw. The one thing a
//! bake needs told is how big an agent is, and those are console variables
//! rather than fields in a file, which is the arrangement `sim.rate` has kept
//! since debt `D4`.

use std::{cmp::Ordering, collections::BinaryHeap};

use super::{Bodies, BodyKind, Meshes, ShapeKind};
use crate::glam::{Mat4, Vec2, Vec3};

/// How wide one cell is unless somebody says otherwise, in world units.
///
/// Half a unit. The erosion below is measured in cells, so a cell much bigger
/// than an agent makes it a cliff rather than a slope; a cell much smaller
/// costs the square of what it buys.
pub const CELL: f32 = 0.5;

/// The smallest cell a bake will use.
///
/// Five centimeters. Below this the ceilings below stop being a safety net and
/// become the thing that decides how much of the world is baked, which is a
/// worse answer than refusing.
pub const MIN_CELL: f32 = 0.05;

/// How far from a wall a walkable cell must be, in world units.
///
/// The radius of the thing that walks. A path is a line, and a line down the
/// middle of a doorway is walkable only by something with no width; taking the
/// walkable set in by the agent's radius is what makes the line mean
/// something. Recast calls this `walkableRadius` and does it the same way,
/// with a distance field over the grid.
pub const RADIUS: f32 = 0.4;

/// The most cells a navmesh may have.
///
/// Two hundred and sixty-two thousand one hundred and forty-four - the ceiling
/// the collision grid keeps, and for the same reason: it is the point past
/// which the array costs more than the walk it saves. At the default cell that
/// is a square about two hundred and fifty units on a side, twice the biggest
/// terrain [`MAX_SIDE`](super::terrain::MAX_SIDE) allows.
///
/// A world bigger than this gets a **coarser cell** rather than a refusal: the
/// cell grows until the grid fits. A navmesh nobody can walk on is a worse
/// answer than a rough one, and [`Navmesh::cell`] says how rough.
pub const MAX_CELLS: usize = 262_144;

/// The most cells on any one axis.
///
/// One thousand and twenty-four, so a world that is a long thin corridor
/// cannot spend its whole budget along one direction - the argument the
/// collision grid's own `MAX_AXIS` makes for a mesh that is a thin sheet.
pub const MAX_AXIS: u32 = 1024;

/// How far above the highest static body a bake's rays start.
///
/// A ray has to begin above everything it might hit, or the thing it misses is
/// the roof of the tallest object in the world.
pub const SKY: f32 = 1.0;

/// What a step to an orthogonal neighbor is worth in the chamfer.
///
/// A `(3, 4)` chamfer: orthogonal three, diagonal four, and a whole cell
/// therefore three. Four thirds is 1.333 against a real diagonal's 1.414,
/// under six percent out - close enough that the eroded edge is a circle
/// rather than the square repeated erosion would leave, and it is two passes
/// over the grid rather than one per cell of radius. Recast's
/// `erodeWalkableArea` is the same arrangement with the same two numbers.
const ORTHOGONAL: u16 = 3;

/// What a diagonal step is worth. @ref [`ORTHOGONAL`].
const DIAGONAL: u16 = 4;

/// What a whole cell of distance is worth. @ref [`ORTHOGONAL`].
const WHOLE: f32 = 3.0;

/// How far ahead one leg of a pulled path may look, in cells.
///
/// A hundred and twenty-eight. The look-ahead is what makes the pull quadratic
/// in the corridor's length, and a leg longer than this is a straight line
/// across half the biggest world the ceiling allows - so the cap costs a
/// corner nobody will see and buys a bound.
const REACH: usize = 128;

/// What a diagonal move costs against an orthogonal one.
///
/// The real square root of two, because this one is a path length rather than
/// a distance field and there is no reason to approximate it.
const CORNER_COST: f32 = core::f32::consts::SQRT_2;

/// The distance a cell nothing has reached still holds.
///
/// Not [`u16::MAX`], because the passes below add a weight to it.
const FAR: u16 = u16::MAX - DIAGONAL;

/// Where each of a cell's eight neighbors is, as `(dx, dz)`.
///
/// The order is the bit order of `Cell::links`, read like text: the three
/// above, the two beside, the three below.
const NEIGHBORS: [(i32, i32); 8] =
	[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Which of [`NEIGHBORS`] are the four orthogonal ones, by index.
///
/// The erosion asks about these and not the diagonals, because Recast's own
/// `rcErodeWalkableArea` does: a cell touching a hole only at a corner is not
/// against a wall in any direction anything walks.
const ORTHOGONALS: [usize; 4] = [1, 3, 4, 6];

/// Which two orthogonal links a diagonal one needs, by [`NEIGHBORS`] index.
///
/// A diagonal step passes through the corner between two orthogonal
/// neighbors, and taking it while either of those is blocked is a character
/// cutting the corner of a wall. Both have to be open - what every grid
/// pathfinder that does not walk through walls checks. `None` is an orthogonal
/// neighbor, which needs nothing.
const CORNERS: [Option<(usize, usize)>; 8] =
	[Some((1, 3)), None, Some((1, 4)), None, None, Some((6, 3)), None, Some((6, 4))];

/// How well a path answered the question.
///
/// The three Fyrox's `PathKind` has, for the reason it has them: a caller that
/// cannot tell a complete path from the nearest one could get will walk
/// something into a wall and never know.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PathKind {
	/// Nowhere to start, nowhere to finish, or nothing baked. The path is
	/// empty.
	#[default]
	None,

	/// The path ends at the standable cell nearest the goal, which is not the
	/// goal. Somewhere on the way is a gap nothing can cross.
	Partial,

	/// The path reaches the goal.
	Full,
}

impl PathKind {
	/// Whether there is a path at all, complete or not.
	#[must_use]
	pub const fn is_some(self) -> bool { !matches!(self, Self::None) }
}

/// How big the thing that walks is, and what it will climb.
///
/// `#[repr(C)]` and `Copy` like every record that crosses the boundary, and
/// **not** `Pod` - the rule the whole boundary keeps, whether or not this one
/// happens to hold a [`Vec3`].
///
/// The two limits are [`character`](super::character)'s own defaults on
/// purpose. A navmesh saying a slope is walkable where
/// [`move_and_slide`](super::character::move_and_slide) will not climb it is a
/// path leading somewhere a character stands at the bottom of, pushing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavSettings {
	/// How wide one cell is, in world units.
	pub cell: f32,

	/// How far from a wall a cell must be to be walkable, in world units.
	pub radius: f32,

	/// The tallest lip between two cells that are still neighbors.
	pub step: f32,

	/// The steepest ground that is still ground, as the cosine of the angle
	/// from straight up. @ref [`character::GROUND`](super::character::GROUND).
	pub slope: f32,
}

impl NavSettings {
	/// The defaults, which are [`character`](super::character)'s.
	pub const DEFAULT: Self = Self {
		cell: CELL,
		radius: RADIUS,
		step: super::character::STEP,
		slope: super::character::GROUND,
	};

	/// The same settings with every number forced into a range a bake can use.
	///
	/// Clamped rather than refused, for the reason
	/// [`Rate::from_hz`](crate::time::Rate::from_hz) is: these arrive from a
	/// console, a console takes what it is typed, and every number a person
	/// can type has to mean something.
	#[must_use]
	pub fn sane(self) -> Self {
		Self {
			cell: if self.cell.is_finite() {
				self.cell.max(MIN_CELL)
			} else {
				CELL
			},
			radius: sane_or(self.radius, RADIUS),
			step: sane_or(self.step, super::character::STEP),
			slope: if self.slope.is_finite() {
				self.slope.clamp(-1.0, 1.0)
			} else {
				super::character::GROUND
			},
		}
	}
}

impl Default for NavSettings {
	fn default() -> Self { Self::DEFAULT }
}

/// What a bake found in one column of the world.
///
/// The whole of what the grid asks about geometry, which is what lets the grid
/// be tested without a solver: a hand-written sampler describing a staircase
/// is a staircase as far as everything here is concerned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column {
	/// How high the standable surface is.
	pub height: f32,

	/// Which way it faces.
	pub normal: Vec3,
}

/// One cell of the grid.
///
/// Six bytes of payload in eight, and there are up to [`MAX_CELLS`] of them -
/// two megabytes at the ceiling, eight kilobytes on the `ground` fixture.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cell {
	/// How high the surface is here. Meaningless unless
	/// [`open`](Self::open).
	height: f32,

	/// Which of the eight [`NEIGHBORS`] can be stepped to from here.
	links: u8,

	/// Whether anything may stand here at all.
	open: bool,
}

/// One cell waiting to be expanded, and what it is guessed to cost.
///
/// Ordered backwards, because [`BinaryHeap`] hands back the greatest and what
/// an A* wants next is the least. Tied estimates break on the cell index, so
/// two runs over the same grid expand the same cells in the same order - the
/// determinism rule the broad phase and the collision grid both keep.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Step {
	/// The cost so far plus the guess at what is left.
	estimate: f32,

	/// Which cell.
	at: usize,
}

impl Eq for Step {}

impl Ord for Step {
	fn cmp(&self, other: &Self) -> Ordering {
		// `total_cmp` rather than `partial_cmp`: an estimate that came out NaN
		// has to order somewhere rather than panic in a comparator.
		other
			.estimate
			.total_cmp(&self.estimate)
			.then_with(|| other.at.cmp(&self.at))
	}
}

impl PartialOrd for Step {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

/// Where a thing that walks may stand, and what is next to what.
///
/// Baked from the collision world by [`build`](Self::build) and asked for
/// paths by [`find_path`](Self::find_path). Host state rather than a record on
/// an entity: a world has exactly one, because a path goes through the whole
/// world at once and two navmeshes that do not know about each other are two
/// worlds. s&box keeps a single `Scene.NavMesh` for that reason; Godot's
/// several regions are stitched edge to edge into one `NavMap`, which is the
/// same statement with more machinery.
#[derive(Clone, Debug, Default)]
pub struct Navmesh {
	/// What it was baked for, after [`NavSettings::sane`].
	settings: NavSettings,

	/// The world-space corner of cell `(0, 0)`, and the height the rays began
	/// at. Only `x` and `z` are read by a query.
	origin: Vec3,

	/// How many cells there are along `x`.
	across: u32,

	/// How many along `z`.
	deep: u32,

	/// The cells, row by row along `x`.
	cells: Vec<Cell>,
}

impl Navmesh {
	/// Nothing baked.
	#[must_use]
	pub fn new() -> Self { Self::default() }

	/// Bakes a grid over a box of the world.
	///
	/// **One column at a time, and the sampler is the only thing that knows
	/// what geometry is.** That is dependency inversion rather than tidiness:
	/// the walkability rules, the erosion and the linking are the whole of
	/// what can be wrong here, and a test that had to stand a solver up to
	/// reach them would be testing the solver instead.
	///
	/// The cell grows until the grid fits [`MAX_CELLS`] and [`MAX_AXIS`], so a
	/// world of any size bakes into something.
	///
	/// @param settings - how big the thing that walks is
	/// @param low - the low corner of the world to bake, in world space
	/// @param high - its high corner
	/// @param sampler - what is standable in one column, given the column's
	/// middle in world space; `None` for a column with nothing to stand on
	/// @return the grid, empty if the box has no extent
	#[must_use]
	pub fn build<Sampler: FnMut(Vec2) -> Option<Column>>(
		settings: NavSettings,
		low: Vec3,
		high: Vec3,
		mut sampler: Sampler,
	) -> Self {
		let settings = settings.sane();
		let span = (high - low).max(Vec3::ZERO);
		let Some((cell, across, deep)) = shape_of(settings.cell, span.x, span.z) else {
			return Self::new();
		};

		let mut grid = Self {
			settings: NavSettings { cell, ..settings },
			origin: Vec3::new(low.x, high.y + SKY, low.z),
			across,
			deep,
			cells: vec![Cell::default(); count(across, deep)],
		};

		for at in 0..grid.cells.len() {
			let (x, z) = split(at, across);
			let Some(column) = sampler(grid.middle(x, z)) else {
				continue;
			};

			// a surface too steep to stand on is not "no surface": the height
			// stays, because the cell beside it still has to know how far down
			// the cliff goes in order to refuse the step. Only `open` is false.
			grid.cells[at] = Cell {
				height: column.height,
				links: 0,
				open: column.normal.y >= grid.settings.slope,
			};
		}

		grid.erode();
		grid.link();
		grid
	}

	/// How wide one cell actually is, which may be more than was asked for.
	#[must_use]
	pub const fn cell(&self) -> f32 { self.settings.cell }

	/// What this was baked for.
	#[must_use]
	pub const fn settings(&self) -> NavSettings { self.settings }

	/// How many cells there are along `x`.
	#[must_use]
	pub const fn across(&self) -> u32 { self.across }

	/// How many cells there are along `z`.
	#[must_use]
	pub const fn deep(&self) -> u32 { self.deep }

	/// How many cells there are in all.
	#[must_use]
	pub fn len(&self) -> usize { self.cells.len() }

	/// Whether nothing was baked.
	#[must_use]
	pub fn is_empty(&self) -> bool { self.cells.is_empty() }

	/// How many cells anything may stand on.
	#[must_use]
	pub fn walkable(&self) -> usize { self.cells.iter().filter(|cell| cell.open).count() }

	/// Whether a cell may be stood on.
	///
	/// @param x - which column
	/// @param z - which row
	#[must_use]
	pub fn is_open(&self, x: u32, z: u32) -> bool { self.at(x, z).is_some_and(|cell| cell.open) }

	/// Where the middle of a cell's surface is, in world space.
	///
	/// @param x - which column
	/// @param z - which row
	/// @return the point, or `None` for a cell outside the grid
	#[must_use]
	pub fn point(&self, x: u32, z: u32) -> Option<Vec3> {
		let cell = self.at(x, z)?;
		let flat = self.middle(x, z);

		Some(Vec3::new(flat.x, cell.height, flat.y))
	}

	/// The nearest cell anything may stand on.
	///
	/// **Nearest in the plane first**, and only then by distance: the cell a
	/// point falls in is the answer whenever it is open at all, and the ring
	/// search below only runs when it is not. That is what stops a character
	/// standing on a hill being sent to the valley floor beside it because the
	/// valley happens to be nearer in a straight line.
	///
	/// @param at - where to ask, in world space
	/// @return the cell, or `None` if nothing in the grid is standable
	#[must_use]
	pub fn nearest(&self, at: Vec3) -> Option<(u32, u32)> {
		let (x, z) = self.clamped(at);

		if self.is_open(x, z) {
			return Some((x, z));
		}

		// outwards a square ring at a time from where the point landed,
		// stopping at the first ring holding anything. Bounded by the grid, so
		// the worst case is a bake with nothing open in it - and there the
		// answer really is that there is nowhere.
		let reach = self.across.max(self.deep);
		let mut best: Option<((u32, u32), f32)> = None;

		for ring in 1..=reach {
			self.ring(at, (x, z), ring, &mut best);

			if best.is_some() {
				break;
			}
		}

		best.map(|(cell, _)| cell)
	}

	/// A path from one point to another.
	///
	/// Two stages. A* over the cells finds a corridor of them, and the funnel
	/// pulls a string through it - without which the path is the corridor's own
	/// cell centers and a character walks the staircase they describe. Three
	/// of the five engines read do this; Fyrox is the one that does not, and
	/// its paths run vertex to vertex.
	///
	/// The corners are **cell centers**, not the two points asked about: a
	/// caller already knows where it is standing, and the last corner is where
	/// the path can actually get to, which is the whole of what
	/// [`PathKind::Partial`] means.
	///
	/// @param from - where the walk starts, in world space
	/// @param to - where it should end
	/// @param into - where to put the corners; cleared first, and the caller
	/// keeps the allocation
	/// @return how well the answer matches the question
	pub fn find_path(&self, from: Vec3, to: Vec3, into: &mut Vec<Vec3>) -> PathKind {
		into.clear();

		let (Some(start), Some(goal)) = (self.nearest(from), self.nearest(to)) else {
			return PathKind::None;
		};
		let Some(corridor) = self.search(start, goal) else {
			return PathKind::None;
		};
		let Some(&last) = corridor.last() else {
			return PathKind::None;
		};

		self.pull(&corridor, into);

		// two conditions, and both are needed. The search may have run out of
		// open cells and returned the nearest it reached; and the goal cell
		// itself may be a long way from the point asked about, which is what
		// happens when somebody asks for a path into a lake.
		if last == place(self.across, goal) && self.reaches(goal, to) {
			PathKind::Full
		} else {
			PathKind::Partial
		}
	}

	/// One cell, if it is in the grid.
	fn at(&self, x: u32, z: u32) -> Option<&Cell> {
		if x >= self.across || z >= self.deep {
			return None;
		}

		self.cells.get(place(self.across, (x, z)))
	}

	/// Where the middle of a cell is in the plane, in world space.
	fn middle(&self, x: u32, z: u32) -> Vec2 {
		Vec2::new(
			(wide(x) + 0.5).mul_add(self.settings.cell, self.origin.x),
			(wide(z) + 0.5).mul_add(self.settings.cell, self.origin.z),
		)
	}

	/// Which cell a world-space point falls in, held inside the grid.
	fn clamped(&self, at: Vec3) -> (u32, u32) {
		let cell = self.settings.cell;
		let along = |value: f32, origin: f32, total: u32| -> u32 {
			cells_in((value - origin) / cell).min(total.saturating_sub(1))
		};

		(along(at.x, self.origin.x, self.across), along(at.z, self.origin.z, self.deep))
	}

	/// Whether a cell is near enough a point to call the path complete.
	///
	/// Within one cell in the plane. A goal in the middle of a lake is not
	/// reached by standing on the shore, and saying so is the whole of what
	/// [`PathKind::Partial`] is for.
	fn reaches(&self, cell: (u32, u32), to: Vec3) -> bool {
		self.middle(cell.0, cell.1)
			.distance(Vec2::new(to.x, to.z))
			<= self.settings.cell
	}

	/// Looks at one square ring of cells for the nearest open one.
	///
	/// @param at - the point being asked about
	/// @param from - the cell it landed in
	/// @param ring - how far out to look, in cells
	/// @param best - the nearest found so far, and how far off it was
	fn ring(&self, at: Vec3, from: (u32, u32), ring: u32, best: &mut Option<((u32, u32), f32)>) {
		let (cx, cz) = (i64::from(from.0), i64::from(from.1));
		let reach = i64::from(ring);

		for step in -reach..=reach {
			// the four sides of the square at once, so the corners are looked
			// at twice and nothing is looked at zero times.
			self.consider(at, (cx + step, cz - reach), best);
			self.consider(at, (cx + step, cz + reach), best);
			self.consider(at, (cx - reach, cz + step), best);
			self.consider(at, (cx + reach, cz + step), best);
		}
	}

	/// Keeps one cell if it is open and nearer than anything so far.
	///
	/// A method rather than the body of the loop above, which is one level of
	/// nesting past what this workspace allows and is the right call anyway.
	///
	/// @param at - the point being asked about
	/// @param cell - where to look, which may be outside the grid
	/// @param best - the nearest found so far, and how far off it was
	fn consider(&self, at: Vec3, cell: (i64, i64), best: &mut Option<((u32, u32), f32)>) {
		let (Ok(x), Ok(z)) = (u32::try_from(cell.0), u32::try_from(cell.1)) else {
			return;
		};
		let Some(point) = self.point(x, z).filter(|_| self.is_open(x, z)) else {
			return;
		};
		let away = point.distance_squared(at);

		if best.is_none_or(|(_, near)| away < near) {
			*best = Some(((x, z), away));
		}
	}

	/// Takes the walkable set in by the agent's radius.
	///
	/// A two-pass chamfer distance to the nearest cell that is not open, and
	/// then a threshold. @ref [`ORTHOGONAL`] for the two weights.
	///
	/// **Everything off the edge of the grid counts as closed**, which is not
	/// a detail: the grid is exactly as big as the static bodies, so its border
	/// is the edge of the world, and a character eroded back from it is one
	/// who does not path off the end of the ground.
	fn erode(&mut self) {
		let reach = self.settings.radius / self.settings.cell * WHOLE;

		if reach <= 0.0 || self.cells.is_empty() {
			return;
		}

		let mut away = self.rim();

		self.sweep(&mut away, true);
		self.sweep(&mut away, false);

		for (cell, distance) in self.cells.iter_mut().zip(away) {
			cell.open &= f32::from(distance) >= reach;
		}
	}

	/// Every cell's distance before the chamfer spreads it.
	///
	/// Nil for a cell nothing may stand on, and **nil for an open cell with a
	/// closed orthogonal neighbor or none at all** - which is the whole of
	/// how the edge of the world gets into a distance field that only ever
	/// looks at cells. Recast's `rcErodeWalkableArea` seeds it the same way,
	/// and accepts the same half-cell of conservatism for it.
	fn rim(&self) -> Vec<u16> {
		(0..self.cells.len())
			.map(|at| if self.walled(at) { 0 } else { FAR })
			.collect()
	}

	/// Whether a cell is against something, the edge of the grid included.
	fn walled(&self, at: usize) -> bool {
		if !self.cells[at].open {
			return true;
		}

		let here = split(at, self.across);

		ORTHOGONALS.into_iter().any(|index| {
			self.offset(here, NEIGHBORS[index])
				.is_none_or(|other| !self.cells[other].open)
		})
	}

	/// One pass of the chamfer, forwards or backwards.
	///
	/// Forwards walks the cells in order and looks at the four neighbors
	/// already visited; backwards is the mirror. Two passes is the whole
	/// transform, which is what a chamfer is for.
	///
	/// @param away - the distances so far
	/// @param forwards - which way to walk
	fn sweep(&self, away: &mut [u16], forwards: bool) {
		let total = self.cells.len();
		let looking: [(i32, i32); 4] = if forwards {
			[(-1, -1), (0, -1), (1, -1), (-1, 0)]
		} else {
			[(1, 1), (0, 1), (-1, 1), (1, 0)]
		};

		for step in 0..total {
			let at = if forwards { step } else { total - 1 - step };

			if self.cells[at].open {
				away[at] = self.reached(away, at, looking);
			} else {
				away[at] = 0;
			}
		}
	}

	/// The shortest distance one cell's already-visited neighbors offer it.
	///
	/// @param away - the distances so far
	/// @param at - the cell being answered
	/// @param looking - which four neighbors this pass has already visited
	fn reached(&self, away: &[u16], at: usize, looking: [(i32, i32); 4]) -> u16 {
		let here = split(at, self.across);
		let mut best = away[at];

		for (dx, dz) in looking {
			let Some(other) = self.offset(here, (dx, dz)) else {
				continue;
			};
			let weight = if dx == 0 || dz == 0 { ORTHOGONAL } else { DIAGONAL };

			best = best.min(away[other].saturating_add(weight));
		}

		best
	}

	/// Works out which cells can be stepped between.
	///
	/// Two cells are neighbors when both are open and the lip between them is
	/// no taller than [`NavSettings::step`] - the same number
	/// [`move_and_slide`](super::character::move_and_slide) climbs, so that the
	/// path and the mover agree about what a stair is. A diagonal also needs
	/// both of the orthogonals it passes between. @ref [`CORNERS`].
	fn link(&mut self) {
		let step = self.settings.step;
		let mut links = vec![0_u8; self.cells.len()];

		for (at, out) in links.iter_mut().enumerate() {
			if self.cells[at].open {
				*out = corners_of(self.stepped(at, step));
			}
		}

		for (cell, mask) in self.cells.iter_mut().zip(links) {
			cell.links = mask;
		}
	}

	/// Which of a cell's eight neighbors are open and near enough in height.
	///
	/// The diagonals here are the raw answer; [`corners_of`] is what takes
	/// back the ones that would cut a wall.
	///
	/// @param at - the cell being answered
	/// @param step - the tallest lip that is still a step
	fn stepped(&self, at: usize, step: f32) -> u8 {
		let here = split(at, self.across);
		let height = self.cells[at].height;
		let mut mask = 0_u8;

		for (index, (dx, dz)) in NEIGHBORS.into_iter().enumerate() {
			let Some(other) = self.offset(here, (dx, dz)) else {
				continue;
			};
			let neighbor = self.cells[other];

			if neighbor.open && (neighbor.height - height).abs() <= step {
				mask |= 1 << index;
			}
		}

		mask
	}

	/// Where a neighbor of a cell is in [`cells`](Self::cells), if it exists.
	fn offset(&self, from: (u32, u32), by: (i32, i32)) -> Option<usize> {
		let x = u32::try_from(i64::from(from.0) + i64::from(by.0)).ok()?;
		let z = u32::try_from(i64::from(from.1) + i64::from(by.1)).ok()?;

		if x >= self.across || z >= self.deep {
			return None;
		}

		Some(place(self.across, (x, z)))
	}

	/// A corridor of cells from one to another, or the nearest it could reach.
	///
	/// A* with an octile heuristic, which is admissible here because a step
	/// costs at least what it covers in the plane and the heuristic is exactly
	/// what it covers in the plane. Cost is the real three-dimensional
	/// distance, so a path over a hill is longer than one around it - which is
	/// what makes a navmesh worth having over a straight line.
	///
	/// @param start - where to begin
	/// @param goal - where to end
	/// @return the cells, start first, or `None` if the start is not standable
	fn search(&self, start: (u32, u32), goal: (u32, u32)) -> Option<Vec<usize>> {
		let (from, to) = (place(self.across, start), place(self.across, goal));

		if !self.cells.get(from)?.open {
			return None;
		}

		let mut came = vec![usize::MAX; self.cells.len()];
		let mut spent = vec![f32::INFINITY; self.cells.len()];
		let mut queue = BinaryHeap::new();
		let mut nearest = (from, self.guess(start, goal));

		spent[from] = 0.0;
		queue.push(Step { estimate: nearest.1, at: from });

		while let Some(Step { at, estimate }) = queue.pop() {
			if at == to {
				return Some(trail(&came, from, to));
			}

			// a cell reached again by a cheaper route is already in the queue
			// twice; the stale copy is the one whose estimate no longer matches
			// what was spent, and dropping it is cheaper than finding it.
			if estimate > spent[at] + self.guess(split(at, self.across), goal) {
				continue;
			}

			self.expand(at, goal, (&mut came, &mut spent, &mut queue), &mut nearest);
		}

		Some(trail(&came, from, nearest.0))
	}

	/// Looks at one cell's neighbors and queues the ones it improves.
	///
	/// @param at - the cell being expanded
	/// @param goal - where the search is heading, for the heuristic
	/// @param search - what came from where, what each cost, and what is queued
	/// @param nearest - the cell closest to the goal so far, and how close
	fn expand(
		&self,
		at: usize,
		goal: (u32, u32),
		search: (&mut [usize], &mut [f32], &mut BinaryHeap<Step>),
		nearest: &mut (usize, f32),
	) {
		let (came, spent, queue) = search;
		let here = split(at, self.across);
		let links = self.cells[at].links;

		for (index, (dx, dz)) in NEIGHBORS.into_iter().enumerate() {
			if links & (1 << index) == 0 {
				continue;
			}

			let Some(other) = self.offset(here, (dx, dz)) else {
				continue;
			};
			let flat = if dx == 0 || dz == 0 { 1.0 } else { CORNER_COST } * self.settings.cell;
			let climb = self.cells[other].height - self.cells[at].height;
			let cost = spent[at] + flat.hypot(climb);

			if cost >= spent[other] {
				continue;
			}

			let left = self.guess(split(other, self.across), goal);

			came[other] = at;
			spent[other] = cost;
			queue.push(Step { estimate: cost + left, at: other });

			if left < nearest.1 {
				*nearest = (other, left);
			}
		}
	}

	/// How far a cell is from the goal, as an A* may guess it.
	///
	/// The octile distance in the plane: the diagonal part costs
	/// [`CORNER_COST`] and the rest costs one. It never over-estimates,
	/// because a step's real cost is its distance in the plane and then some.
	fn guess(&self, from: (u32, u32), to: (u32, u32)) -> f32 {
		let dx = wide(from.0.abs_diff(to.0));
		let dz = wide(from.1.abs_diff(to.1));
		let (long, short) = (dx.max(dz), dx.min(dz));

		short.mul_add(CORNER_COST, long - short) * self.settings.cell
	}

	/// Pulls a straight string through a corridor of cells.
	///
	/// **Line of sight rather than a funnel, and that is a measurement rather
	/// than a preference.** The funnel is what a polygonal navmesh uses and is
	/// what this had first: the corridor's shared edges are portals, the two
	/// sides close in, and where they cross is a corner. On a grid it
	/// degenerates, because the straight line between two cell centers passes
	/// exactly through the point where two consecutive portals meet - so the
	/// sides meet there too, the funnel restarts, and a corner comes out. A
	/// plain diagonal across an empty floor came back as **eighteen corners
	/// over eighteen cells**, which is the corridor with extra steps.
	///
	/// What works on a grid is the other family: keep the farthest cell that
	/// can still be walked to in a straight line, make that a corner, and go
	/// again. No degenerate case, half the code, and the same answer on
	/// everything the funnel got right.
	///
	/// @param corridor - the cells, start first
	/// @param into - where to put the corners
	fn pull(&self, corridor: &[usize], into: &mut Vec<Vec3>) {
		let Some(&first) = corridor.first() else {
			return;
		};

		into.push(self.surface(first));

		let mut anchor = 0;

		while anchor + 1 < corridor.len() {
			let reach = self.farthest(corridor, anchor);

			into.push(self.surface(corridor[reach]));
			anchor = reach;
		}
	}

	/// The farthest cell of a corridor still reachable in a straight line.
	///
	/// **Stops at the first cell it cannot see rather than looking past it**,
	/// which is what every grid smoother does: visibility over a grid is not
	/// monotone, and a leg that skipped a blocked cell to find a clear one
	/// behind it would be a leg through the blocked one.
	///
	/// @param corridor - the cells, start first
	/// @param anchor - where this leg begins
	/// @return the index the leg ends at, always past `anchor`
	fn farthest(&self, corridor: &[usize], anchor: usize) -> usize {
		let mut reach = anchor + 1;

		for step in (anchor + 2)..(anchor + REACH).min(corridor.len()) {
			if !self.sees(corridor[anchor], corridor[step]) {
				break;
			}

			reach = step;
		}

		reach
	}

	/// Whether a straight walk between two cells stays on ground.
	///
	/// The line is walked one axis at a time, so what it tests is an
	/// orthogonal staircase between the two rather than the geometric segment.
	/// **That is conservative on purpose**: every step it takes is a link the
	/// bake put there, so a leg it approves is one somebody can really walk -
	/// height rule, corner rule and all. It can refuse a leg a geometric test
	/// would have allowed, which costs a corner and never costs a wall.
	///
	/// @param from - where the leg starts
	/// @param to - where it would end
	fn sees(&self, from: usize, to: usize) -> bool {
		let (start, goal) = (split(from, self.across), split(to, self.across));
		let (mut x, mut z) = (i64::from(start.0), i64::from(start.1));
		let (tx, tz) = (i64::from(goal.0), i64::from(goal.1));
		let (wide, deep) = ((tx - x).abs(), (tz - z).abs());
		let (along_x, along_z) = ((tx - x).signum(), (tz - z).signum());
		let mut error = wide - deep;

		for _ in 0..=(wide + deep) {
			if (x, z) == (tx, tz) {
				return true;
			}

			// Bresenham's choice, but taking one axis at a time rather than
			// both at once, so the walk never cuts through the corner between
			// two cells it did not visit.
			let sideways = error * 2 > -deep;
			let step = if sideways { (along_x, 0) } else { (0, along_z) };
			let (Ok(here_x), Ok(here_z)) = (u32::try_from(x), u32::try_from(z)) else {
				return false;
			};

			if !self.links((here_x, here_z), step) {
				return false;
			}

			if sideways {
				error -= deep;
				x += along_x;
			} else {
				error += wide;
				z += along_z;
			}
		}

		false
	}

	/// Whether one cell links to the neighbor in a direction.
	///
	/// @param from - which cell
	/// @param step - which way, as one of [`NEIGHBORS`]
	fn links(&self, from: (u32, u32), step: (i64, i64)) -> bool {
		let Some(index) = NEIGHBORS
			.iter()
			.position(|&(dx, dz)| i64::from(dx) == step.0 && i64::from(dz) == step.1)
		else {
			return false;
		};

		self.at(from.0, from.1)
			.is_some_and(|cell| cell.links & (1 << index) != 0)
	}

	/// Where a cell's surface is, by its place in the flat array.
	fn surface(&self, at: usize) -> Vec3 {
		let (x, z) = split(at, self.across);

		self.point(x, z).unwrap_or(self.origin)
	}
}

/// Takes back every diagonal link whose two orthogonals are not both there.
///
/// @param mask - the eight links as [`Navmesh::stepped`] worked them out
/// @return the same, with the corner-cutting diagonals gone
fn corners_of(mask: u8) -> u8 {
	let mut out = mask;

	for (index, corner) in CORNERS.into_iter().enumerate() {
		let Some((first, second)) = corner else {
			continue;
		};

		if mask & (1 << first) == 0 || mask & (1 << second) == 0 {
			out &= !(1 << index);
		}
	}

	out
}

/// Walks the trail of what came from where, back to front.
///
/// @param came - one entry a cell, `usize::MAX` for a cell nothing reached
/// @param from - where the search started
/// @param to - where to walk back from
/// @return the cells from `from` to `to`
fn trail(came: &[usize], from: usize, to: usize) -> Vec<usize> {
	let mut path = vec![to];
	let mut at = to;

	// bounded by the number of cells, because every entry was written once and
	// a cycle would need one to be written twice.
	for _ in 0..came.len() {
		if at == from {
			break;
		}

		let Some(&before) = came
			.get(at)
			.filter(|&&before| before != usize::MAX)
		else {
			break;
		};

		path.push(before);
		at = before;
	}

	path.reverse();
	path
}

/// How many cells a bake gets on each axis, and how wide one is.
///
/// The cell grows rather than the grid being refused. @ref [`MAX_CELLS`].
///
/// @param asked - the cell size the settings wanted
/// @param wide - how far the box reaches along `x`
/// @param deep - how far along `z`
/// @return `(cell, across, deep)`, or `None` for a box with no extent
fn shape_of(asked: f32, wide: f32, deep: f32) -> Option<(f32, u32, u32)> {
	if !(wide > 0.0 && deep > 0.0) {
		return None;
	}

	let mut cell = asked.max(MIN_CELL);

	// doubling rather than solving for it: the answer is the same and there is
	// no square root to get wrong. At most a few dozen turns, because each one
	// quarters the count.
	for _ in 0..64 {
		let across = spread(wide, cell);
		let down = spread(deep, cell);

		if across <= MAX_AXIS && down <= MAX_AXIS && count(across, down) <= MAX_CELLS {
			return Some((cell, across, down));
		}

		cell *= 2.0;
	}

	None
}

/// How many cells of a size cover a span, at least one.
fn spread(span: f32, cell: f32) -> u32 { cells_in((span / cell).ceil()).clamp(1, MAX_AXIS) }

/// How many cells a grid of a shape has.
fn count(across: u32, deep: u32) -> usize {
	usize::try_from(u64::from(across) * u64::from(deep)).unwrap_or(MAX_CELLS)
}

/// Where a cell is in the flat array.
fn place(across: u32, at: (u32, u32)) -> usize {
	usize::try_from(u64::from(at.1) * u64::from(across) + u64::from(at.0)).unwrap_or(0)
}

/// Which cell an index in the flat array is.
fn split(at: usize, across: u32) -> (u32, u32) {
	let across = usize::try_from(across.max(1)).unwrap_or(1);
	let x = u32::try_from(at % across).unwrap_or(0);
	let z = u32::try_from(at / across).unwrap_or(0);

	(x, z)
}

/// A count of cells as a float.
///
/// Through a `u16`, which is what [`terrain`](super::terrain)'s own helper does
/// and for the same reason: this workspace refuses `as`, and [`MAX_AXIS`] is
/// one thousand and twenty-four.
fn wide(count: u32) -> f32 { f32::from(u16::try_from(count).unwrap_or(u16::MAX)) }

/// A float as a count of cells, with everything below nil counting as none.
///
/// @param value - how many cells, as arithmetic left it
fn cells_in(value: f32) -> u32 {
	if value
		.partial_cmp(&1.0)
		.is_none_or(Ordering::is_lt)
	{
		return 0;
	}

	let held = value.min(f32::from(u16::MAX));

	#[expect(
		clippy::as_conversions,
		clippy::cast_possible_truncation,
		clippy::cast_sign_loss,
		reason = "clamped to a span a u16 holds exactly, and the fraction is meant to go"
	)]
	let whole = held as u32;

	whole
}

/// The smallest box holding everything a thing that walks could stand on.
///
/// **Static bodies only.** What a navmesh describes is where the world lets
/// somebody stand, and a crate that is going to roll away is not the world -
/// s&box's baker takes a whole `PhysicsWorld` and Unreal reads every
/// `UBodySetup`, but both of them rebuild when the geometry moves, and colby's
/// answer to the same question is to bake what does not move. A dynamic body
/// still blocks the column it is in at the moment of the bake, which is the
/// right answer for a crate that has settled and a stale one for a crate that
/// has not - and that is a rebuild, not a different box.
///
/// @param bodies - the table to measure
/// @param meshes - the registry, for the bodies that are meshes
/// @return `(low, high)`, or `None` for a world with no static body in it
#[must_use]
pub fn ground_bounds(bodies: &Bodies, meshes: &Meshes) -> Option<(Vec3, Vec3)> {
	let mut low = Vec3::splat(f32::INFINITY);
	let mut high = Vec3::splat(f32::NEG_INFINITY);
	let mut found = false;

	for (_, body) in bodies.iter() {
		if body.kind != BodyKind::Static {
			continue;
		}

		let Some((near, far)) = around(body.bounds(), body, meshes) else {
			continue;
		};

		low = low.min(near);
		high = high.max(far);
		found = true;
	}

	found.then_some((low, high))
}

/// One body's world-space box, mesh or otherwise.
///
/// A mesh body has no bounds of its own here - the geometry is in the registry
/// rather than in the record - so this is the one place that goes and gets it.
/// The eight corners through the model matrix and the box around where they
/// land, which is what the solver's own `world_bounds` does with its baked
/// copy.
///
/// @param plain - what the body could answer on its own
/// @param body - the body
/// @param meshes - the registry
fn around(
	plain: Option<(Vec3, Vec3)>,
	body: &super::Body,
	meshes: &Meshes,
) -> Option<(Vec3, Vec3)> {
	if let Some(bounds) = plain {
		return Some(bounds);
	}

	if body.shape.kind != ShapeKind::Mesh {
		return None;
	}

	let mesh = meshes.get(body.shape.mesh)?.value();

	if mesh.is_empty() {
		return None;
	}

	let (near, far) = mesh.bounds();

	Some(placed(&body.transform.matrix(), near, far))
}

/// A box through a transform, made axis-aligned again.
fn placed(matrix: &Mat4, low: Vec3, high: Vec3) -> (Vec3, Vec3) {
	let mut near = Vec3::splat(f32::INFINITY);
	let mut far = Vec3::splat(f32::NEG_INFINITY);

	for index in 0..8_u32 {
		let corner = Vec3::new(
			if index & 1 == 0 { low.x } else { high.x },
			if index & 2 == 0 { low.y } else { high.y },
			if index & 4 == 0 { low.z } else { high.z },
		);
		let put = matrix.transform_point3(corner);

		near = near.min(put);
		far = far.max(put);
	}

	(near, far)
}

/// A number a bake can use, or a default for one it cannot.
///
/// @param value - what was asked for
/// @param fallback - what to use for a number that is not finite
fn sane_or(value: f32, fallback: f32) -> f32 {
	if value.is_finite() {
		value.clamp(0.0, 64.0)
	} else {
		fallback
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A world twenty units square with nothing in it but flat ground.
	const LOW: Vec3 = Vec3::new(-10.0, 0.0, -10.0);

	/// The other corner of it.
	const HIGH: Vec3 = Vec3::new(10.0, 0.0, 10.0);

	/// Settings that erode nothing, so a test about linking is about linking.
	const BARE: NavSettings = NavSettings {
		radius: 0.0,
		cell: 1.0,
		..NavSettings::DEFAULT
	};

	/// A sampler describing an endless floor at nil.
	fn floor(_: Vec2) -> Option<Column> { Some(Column { height: 0.0, normal: Vec3::Y }) }

	/// A grid over the world above, with a sampler and settings.
	fn baked<Sampler: FnMut(Vec2) -> Option<Column>>(
		settings: NavSettings,
		sampler: Sampler,
	) -> Navmesh {
		Navmesh::build(settings, LOW, HIGH, sampler)
	}

	/// Where a path between two points goes, and how well it answered.
	fn walked(grid: &Navmesh, from: Vec3, to: Vec3) -> (PathKind, Vec<Vec3>) {
		let mut path = Vec::new();
		let kind = grid.find_path(from, to, &mut path);

		(kind, path)
	}

	#[test]
	fn a_flat_world_is_walkable_everywhere_but_its_rim() {
		let grid = baked(NavSettings { cell: 1.0, ..NavSettings::DEFAULT }, floor);

		assert_eq!((grid.across(), grid.deep()), (20, 20), "twenty cells of one unit");
		assert!(grid.is_open(10, 10), "the middle of a floor is standable");

		// the rim is not: the grid is exactly as big as the world, so its edge
		// is the edge of the ground and an agent with a radius does not stand
		// with half of itself over the drop.
		assert!(!grid.is_open(0, 0), "and the corner of it is not");
		assert!(!grid.is_open(0, 10), "nor the middle of a side");
	}

	#[test]
	fn erosion_takes_the_walkable_set_in_by_the_agents_radius() {
		let wide = baked(
			NavSettings {
				radius: 0.0,
				cell: 1.0,
				..NavSettings::DEFAULT
			},
			floor,
		);
		let narrow = baked(
			NavSettings {
				radius: 2.5,
				cell: 1.0,
				..NavSettings::DEFAULT
			},
			floor,
		);

		assert_eq!(wide.walkable(), 400, "nothing eroded is the whole floor");
		assert!(narrow.walkable() < wide.walkable(), "and a radius costs the rim");

		// two and a half cells in, near enough: the chamfer is within six
		// percent of a real distance and the threshold is one comparison.
		assert!(!narrow.is_open(2, 10), "two cells from the edge is still too near");
		assert!(narrow.is_open(3, 10), "three is far enough");
	}

	#[test]
	fn a_slope_too_steep_to_stand_on_is_not_walkable() {
		// a cliff down the middle: everything past the tenth cell faces
		// sideways rather than up.
		let grid = baked(BARE, |at| {
			let normal = if at.x > 0.0 {
				Vec3::new(0.9, 0.436, 0.0).normalize()
			} else {
				Vec3::Y
			};

			Some(Column { height: 0.0, normal })
		});

		assert!(grid.is_open(5, 10), "flat ground is ground");
		assert!(!grid.is_open(15, 10), "a sixty-four degree face is not");
	}

	#[test]
	fn a_lip_taller_than_a_step_is_not_a_neighbor() {
		let tall = NavSettings { step: 0.35, ..BARE };
		let grid = baked(tall, |at| {
			let height = if at.x > 0.0 { 0.4 } else { 0.0 };

			Some(Column { height, normal: Vec3::Y })
		});
		let (kind, _) = walked(&grid, Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.4, 0.0));

		assert_eq!(kind, PathKind::Partial, "four tenths is not a stair of three and a half");

		// and the same wall a hair lower is one.
		let low = baked(tall, |at| {
			let height = if at.x > 0.0 { 0.3 } else { 0.0 };

			Some(Column { height, normal: Vec3::Y })
		});
		let (climbed, _) = walked(&low, Vec3::new(-5.0, 0.0, 0.0), Vec3::new(5.0, 0.3, 0.0));

		assert_eq!(climbed, PathKind::Full, "three tenths is");
	}

	#[test]
	fn a_diagonal_through_the_corner_of_a_wall_is_refused() {
		// two blocked cells meeting at a corner. The only straight way past
		// them is between the two, which is a character walking through a wall.
		let grid = baked(BARE, |at| {
			let blocked = (at.x > 0.0 && at.x < 1.0 && at.y > -1.0 && at.y < 0.0)
				|| (at.x > -1.0 && at.x < 0.0 && at.y > 0.0 && at.y < 1.0);

			(!blocked).then_some(Column { height: 0.0, normal: Vec3::Y })
		});

		assert!(grid.is_open(10, 10) && grid.is_open(9, 9), "the two are open");
		assert!(!grid.is_open(10, 9) && !grid.is_open(9, 10), "and the corner is not");

		let mut path = Vec::new();
		let across = grid.find_path(
			grid.point(9, 9).expect("in the grid"),
			grid.point(10, 10).expect("in the grid"),
			&mut path,
		);

		assert_eq!(across, PathKind::Full, "there is a way round");
		assert!(path.len() > 2, "and it is not the diagonal: {path:?}");
	}

	#[test]
	fn the_funnel_pulls_a_straight_corridor_straight() {
		let grid = baked(BARE, floor);
		let (kind, path) = walked(&grid, Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));

		assert_eq!(kind, PathKind::Full);
		assert_eq!(path.len(), 2, "a line across an empty floor has two ends: {path:?}");

		// half a cell off the line asked for, because a corner is a cell's
		// middle and the line runs along a cell boundary. What matters is that
		// the two ends agree, which is what straight means here.
		let wander = path[0].z - path[1].z;

		assert!(wander.abs() < 1.0e-3, "and it does not wander: {path:?}");
		assert!(path[0].z.abs() <= grid.cell().mul_add(0.5, 1.0e-3), "nor drift: {path:?}");
	}

	#[test]
	fn a_wall_with_a_gap_is_walked_round_rather_than_through() {
		let grid = baked(BARE, |at| {
			let wall = at.x.abs() < 1.0 && at.y < 6.0;

			(!wall).then_some(Column { height: 0.0, normal: Vec3::Y })
		});
		let (kind, path) = walked(&grid, Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));

		assert_eq!(kind, PathKind::Full, "the doorway is open");
		assert!(path.len() >= 3, "a way round has a corner in it: {path:?}");
		assert!(
			path.iter().any(|point| point.z > 5.0),
			"and the corner is at the doorway: {path:?}"
		);

		// every corner of the answer is somewhere a thing may stand, which is
		// the property a funnel is easiest to get wrong in.
		for point in &path {
			let cell = grid.nearest(*point).expect("somewhere");
			let middle = grid.point(cell.0, cell.1).expect("in the grid");

			assert!(
				middle.distance(*point) <= grid.cell() * CORNER_COST,
				"{point:?} is not on the mesh"
			);
		}
	}

	#[test]
	fn a_goal_nothing_can_reach_gives_the_nearest_it_could() {
		let grid = baked(BARE, |at| {
			let wall = at.x.abs() < 1.0;

			(!wall).then_some(Column { height: 0.0, normal: Vec3::Y })
		});
		let (kind, path) = walked(&grid, Vec3::new(-6.0, 0.0, 0.0), Vec3::new(6.0, 0.0, 0.0));

		assert_eq!(kind, PathKind::Partial, "the other side is another world");
		assert!(!path.is_empty(), "and the answer is still where it could get to");

		for point in &path {
			assert!(point.x < 0.0, "which is all on this side: {point:?}");
		}
	}

	#[test]
	fn a_world_with_nothing_to_stand_on_has_no_paths() {
		let grid = baked(BARE, |_| None);

		assert_eq!(grid.walkable(), 0, "nothing was standable");

		let (kind, path) = walked(&grid, Vec3::ZERO, Vec3::new(5.0, 0.0, 5.0));

		assert_eq!(kind, PathKind::None);
		assert!(path.is_empty());
	}

	#[test]
	fn a_bake_with_no_extent_is_empty_rather_than_wrong() {
		let flat = Navmesh::build(BARE, Vec3::ZERO, Vec3::ZERO, floor);

		assert!(flat.is_empty(), "a point is not a world");
		assert_eq!(flat.walkable(), 0);
		assert_eq!(walked(&flat, Vec3::ZERO, Vec3::ONE).0, PathKind::None);
	}

	#[test]
	fn a_world_too_big_for_the_ceiling_gets_a_coarser_cell() {
		let huge = Vec3::new(4000.0, 0.0, 4000.0);
		let grid = Navmesh::build(NavSettings { cell: 0.1, ..BARE }, Vec3::ZERO, huge, floor);

		assert!(grid.cell() > 0.1, "the cell grew rather than the bake being refused");
		assert!(grid.len() <= MAX_CELLS, "and it grew until the grid fit: {}", grid.len());
		assert!(grid.across() <= MAX_AXIS && grid.deep() <= MAX_AXIS);
	}

	#[test]
	fn a_number_a_console_can_type_always_means_something() {
		let nonsense = NavSettings {
			cell: f32::NAN,
			radius: f32::INFINITY,
			step: -4.0,
			slope: 40.0,
		}
		.sane();

		assert!((nonsense.cell - CELL).abs() < f32::EPSILON, "not a number is the default");
		assert!((nonsense.radius - RADIUS).abs() < f32::EPSILON, "and so is an infinity");
		assert!(nonsense.step.abs() < f32::EPSILON, "a negative clamps rather than refusing");
		assert!((nonsense.slope - 1.0).abs() < f32::EPSILON, "a cosine is a cosine");
	}

	#[test]
	fn the_same_grid_answers_the_same_question_the_same_way() {
		// the property `--link` cannot see and every other spatial structure in
		// this engine keeps: two runs expand the same cells in the same order,
		// so a path is a fact about the world rather than about the heap's mood.
		let grid = baked(BARE, |at| {
			let wall = at.x.abs() < 1.0 && at.y < 4.0;

			(!wall).then_some(Column { height: 0.0, normal: Vec3::Y })
		});
		let (first, once) = walked(&grid, Vec3::new(-7.0, 0.0, -7.0), Vec3::new(7.0, 0.0, 7.0));
		let (second, again) = walked(&grid, Vec3::new(-7.0, 0.0, -7.0), Vec3::new(7.0, 0.0, 7.0));

		assert_eq!(first, second);
		assert_eq!(once, again, "and the same corners, not merely the same length");
	}

	#[test]
	fn a_point_off_the_edge_of_the_world_still_finds_a_cell() {
		let grid = baked(BARE, floor);

		assert!(
			grid.nearest(Vec3::new(-500.0, 0.0, 500.0))
				.is_some(),
			"the nearest standable cell, not a refusal"
		);

		let (kind, path) = walked(&grid, Vec3::new(-500.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0));

		assert_eq!(kind, PathKind::Full, "and a path from it is a path");
		assert!(!path.is_empty());
	}

	#[test]
	fn a_path_climbs_a_staircase_it_could_not_jump() {
		// four treads of three tenths each, which is under `step` one at a time
		// and well over it end to end.
		let grid = baked(BARE, |at| {
			let tread = ((at.x + 10.0) / 5.0).floor().clamp(0.0, 3.0);

			Some(Column { height: tread * 0.3, normal: Vec3::Y })
		});
		let (kind, path) = walked(&grid, Vec3::new(-8.0, 0.0, 0.0), Vec3::new(8.0, 0.9, 0.0));

		assert_eq!(kind, PathKind::Full, "one tread at a time is a climb");
		assert!(
			path.last().is_some_and(|point| point.y > 0.5),
			"and the far end is at the top: {path:?}"
		);
	}
}
