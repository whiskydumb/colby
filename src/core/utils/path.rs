//! Path arithmetic that does not touch the filesystem.
//!
//! One function, and it lives here rather than beside either of its callers
//! because both ask the same question about a path that may not be there. The
//! asset compiler resolves what a source names before the file exists; the
//! launcher compares two spellings of one project directory, and either may
//! lead through a directory nobody ever made.
//!
//! `fs::canonicalize` answers neither. It is the filesystem's answer, it
//! resolves links, and on unix it fails outright on a path that steps through
//! something that is not there - so `yard/../garden` is a path it can say
//! nothing about while Windows folds the step away and answers. Two platforms,
//! two answers, and the one written here is the same on both.

use std::path::{Component, Path, PathBuf};

/// Resolves `.` and `..` without touching the filesystem.
///
/// **A step up over nothing stays a step up.** A tree named `../project` has
/// to keep its shape when something is compared against it; folding that step
/// away on one side of a comparison is what makes two spellings of one
/// directory look like two directories.
///
/// @param path - the path to fold
/// @return the same path with every `.` dropped and every `..` taken against
/// the name before it
#[must_use]
pub fn lexical(path: &Path) -> PathBuf {
	let mut out = PathBuf::new();

	for part in path.components() {
		match part {
			| Component::CurDir => {},
			| Component::ParentDir => step_up(&mut out),
			| other => out.push(other),
		}
	}

	out
}

/// A step up: over a name it takes the name away, over nothing or over another
/// step up it stays a step up.
fn step_up(path: &mut PathBuf) {
	if matches!(path.components().next_back(), Some(Component::Normal(_))) {
		path.pop();
	} else {
		path.push("..");
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_step_that_goes_nowhere_is_dropped_and_a_step_up_takes_a_name() {
		assert_eq!(lexical(Path::new("assets/./meshes")), Path::new("assets/meshes"));
		assert_eq!(lexical(Path::new("assets/ui/../meshes")), Path::new("assets/meshes"));
		assert_eq!(
			lexical(Path::new("projects/other/../yard")),
			Path::new("projects/yard"),
			"the step up is taken whether or not anything named `other` is there"
		);
	}

	#[test]
	fn a_step_up_over_nothing_stays_a_step_up() {
		assert_eq!(lexical(Path::new("../project/assets")), Path::new("../project/assets"));
		assert_eq!(
			lexical(Path::new("../../one/../two")),
			Path::new("../../two"),
			"and one that climbs twice keeps both steps"
		);
	}
}
