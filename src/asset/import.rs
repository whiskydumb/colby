//! The sidecar a source may be imported with: `lamp.gltf.model`.
//!
//! **A file in the source tree that is an input and not a source.** It has no
//! output of its own; what it does is change what the file beside it compiles
//! into. A stylesheet is the same shape - @ref
//! [`document::stylesheets`](crate::document::stylesheets) - and this is the
//! second one.
//!
//! **Its name is the source's whole name plus `.model`**, so `lamp.gltf.model`
//! and `lamp.obj.model` are two different files. That is Godot's spelling
//! (`icon.svg.import`) and it is not decoration: `lamp.obj` and `lamp.gltf`
//! may stand in one directory, because they compile to different outputs, and
//! a `lamp.model` between them would belong to neither.
//!
//! ```json
//! {
//!   "scale": [0.01, 0.01, 0.01],
//!   "skip": ["collision_proxy"],
//!   "materials": { "brass": "materials/brass" }
//! }
//! ```
//!
//! **Three things are decided here**, and they are the three the field
//! actually uses. Counted over the 547 `.vmdl` in s&box's tree: 1905
//! `RenderMeshFile` with an import transform on each, 510 files with a
//! `MaterialGroupList`, and an `import_filter` on the meshes; against 71 with
//! any physics and 30 with any animation. Unreal is the same three under
//! `ImportOffsetTranslation` / `ImportOffsetRotation` /
//! `ImportOffsetUniformScale` and its material slots.
//!
//! - **Where it lands**, as a [`Transform`] spelled the way a `.scene` spells
//!   one, because a second spelling of a placement would be a second thing to
//!   keep in step.
//! - **What not to take**, as a list of the node names to leave out. A piece
//!   nothing stands any more is not written at all, so hiding a collision proxy
//!   saves its `.cmesh` too.
//! - **What a surface is made of**, as a map from the name the file gave a
//!   material to the asset name of one somebody wrote. That is s&box's
//!   `DefaultMaterialGroup.remaps { from, to }`, and it is how a `.material`
//!   from step 5e gets worn without a gesture in the editor.
//!
//! **And a fourth, how coarsely a mesh may be drawn from far away**, as the
//! most coarser levels each mesh of the file is compiled with. Having none is
//! the ordinary case here too: every mesh is compiled with as many levels as it
//! thins out to, @ref [`crate::simplify`], and the key is for the file that
//! wants fewer, or none - `"levels": 0` is a mesh that is only ever drawn
//! whole.
//!
//! **A `.obj` takes two of the four.** It compiles to a `.cmesh`, which is
//! one mesh with no names and no materials in it, so a sidecar beside one that
//! says `skip` or `materials` is refused rather than half-obeyed. The
//! transform is meaningful and is the one knob an OBJ most wants, the format
//! carrying no unit at all: Unreal runs `.obj` through the same Interchange
//! pipeline as everything else and hands it the same three offsets. The levels
//! mean what they mean for any other mesh.
//!
//! **Not here, and each for a reason**: attachment points, which nothing in
//! this engine has; a physics shape, which a `.cmodel` has nowhere to put; clip
//! trimming, which is a card of its own; and renaming a bone, which s&box does
//! 288 times in its tree and Unreal does not do at import at all - it retargets
//! afterwards, with an asset.
//!
//! **Having none is the ordinary case**, and that is a deliberate difference
//! from all three references: s&box's `.vmdl` *is* the asset, Unreal keeps the
//! settings inside the `.uasset`, and Godot's editor writes an `.import`
//! beside every source whether or not anything in it moved. None of the three
//! has a "no settings file" case. colby has no import dialog between dropping
//! a `.gltf` into `assets/` and its compiling - the compiler runs on a timer,
//! four times a second - so the file has to be the exception rather than the
//! rule.

use std::path::{Path, PathBuf};

use colby_core::{
	Result,
	abi::{MAX_LEVELS, Transform},
	err,
	glam::Mat4,
};

use crate::{
	json::Value,
	level::{Rows, Writing, as_text, check, names, put_all, read},
};

/// The extension a sidecar is written with, after the source's whole name.
pub const EXTENSION: &str = "model";

/// How far a composed transform may drift before it is called a shear.
///
/// The same number [`geometry`](crate::gltf) uses for the same question, and
/// deliberately: a placement folded under an import transform is the same
/// arithmetic as a node folded under its parent.
const SQUARE_ENOUGH: f32 = 1e-4;

/// The keys read by hand beside the transform's table: two lists of names and
/// a count.
const HAND: [&str; 3] = ["skip", "materials", "levels"];

/// What a sidecar says.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Import {
	/// Where what the file holds lands, applied outside everything in it.
	pub transform: Transform,

	/// The node names to leave out, in the order they were written.
	pub skip: Vec<String>,

	/// What each of the file's own materials is really made of: the name the
	/// file gave it, and the asset name to wear instead.
	pub materials: Vec<(String, String)>,

	/// The most coarser levels each mesh is compiled with, or nothing for as
	/// many as it thins out to. @ref [`Import::levels`].
	pub levels: Option<usize>,
}

impl Import {
	/// A sidecar that changes nothing, which is what having none means.
	pub const NONE: Self = Self {
		transform: Transform::IDENTITY,
		skip: Vec::new(),
		materials: Vec::new(),
		levels: None,
	};

	/// How many coarser levels a mesh of this source may be compiled with.
	///
	/// [`MAX_LEVELS`] when the sidecar does not say, which is what a source
	/// with no sidecar gets too; a mesh stops short of it wherever it stops
	/// thinning out.
	#[must_use]
	pub fn levels(&self) -> usize { self.levels.unwrap_or(MAX_LEVELS).min(MAX_LEVELS) }

	/// Whether this one says anything at all.
	///
	/// A file holding `{}` says nothing and is the same as no file - except
	/// that it exists, which is what makes the output's mark honest.
	#[must_use]
	pub fn is_silent(&self) -> bool { *self == Self::NONE }

	/// Whether a node by this name is taken.
	///
	/// @param name - the node's name, as the file gave it
	#[must_use]
	pub fn takes(&self, name: &str) -> bool { !self.skip.iter().any(|it| it == name) }

	/// What a material of the file's is really made of, when something else
	/// is.
	///
	/// @param name - the name the file gave it
	#[must_use]
	pub fn material(&self, name: &str) -> Option<&str> {
		self.materials
			.iter()
			.find(|(from, _)| from == name)
			.map(|(_, to)| to.as_str())
	}

	/// One placement of the file, moved to where this says it lands.
	///
	/// Composed as matrices rather than field by field, because an import
	/// transform is the parent of everything in the file and a turned child
	/// under an uneven scale is not a translation, a rotation and a scale.
	/// What comes back is the nearest thing that is not a shear, and how far
	/// that was is the second half of the answer.
	///
	/// @param placed - where the file put it
	/// @return where it lands, and how far the recomposition drifted
	#[must_use]
	pub fn applied(&self, placed: Transform) -> (Transform, f32) {
		if self.transform == Transform::IDENTITY {
			return (placed, 0.0);
		}

		let world = self.transform.matrix() * placed.matrix();
		let landed = Transform::from_matrix(world);
		let drift = landed
			.matrix()
			.to_cols_array()
			.iter()
			.zip(world.to_cols_array())
			.map(|(made, was)| (made - was).abs())
			.fold(0.0_f32, f32::max);

		(landed, drift)
	}

	/// The matrix this puts a mesh's own vertices through.
	#[must_use]
	pub fn matrix(&self) -> Mat4 { self.transform.matrix() }
}

/// Where the sidecar of a source would be, whether or not there is one.
///
/// @param source - the file it would accompany
#[must_use]
pub fn beside(source: &Path) -> PathBuf {
	let name = source
		.file_name()
		.map(|name| name.to_string_lossy().into_owned())
		.unwrap_or_default();

	source.with_file_name(format!("{name}.{EXTENSION}"))
}

/// The source file an asset name came from, if one is in the tree.
///
/// The other direction from the compiler's, which goes from a file to a name.
/// A panel that wants to show what guided a model has only the name, and which
/// of the three extensions the file wears is a question about the tree rather
/// than about the name - so it is asked here, once, in the order the compiler
/// would have found them.
///
/// @param assets - the source tree, `<project>/assets`
/// @param name - the asset name, `models/lamp`
/// @return the file, or nothing when no source under that name is there
#[must_use]
pub fn source_of(assets: &Path, name: &str) -> Option<PathBuf> {
	let stem = name
		.split('/')
		.fold(assets.to_path_buf(), |path, part| path.join(part));

	[crate::gltf::EXTENSION, crate::gltf::BINARY_EXTENSION, crate::obj::EXTENSION]
		.into_iter()
		.map(|extension| stem.with_extension(extension))
		.find(|path| path.is_file())
}

/// The sidecar of a source, if it has one.
///
/// Nothing rather than a default when the file is not there, so that a caller
/// can tell "says nothing" from "is not there" - the first is a file somebody
/// emptied and the second is the ordinary case, and the compiled output
/// records which. @ref [`Import::is_silent`].
///
/// @param source - the `.gltf`, `.glb` or `.obj`
/// @return what it says, or nothing when there is no file, or why the file
/// could not be used
pub fn read_beside(source: &Path) -> Result<Option<Import>> {
	let path = beside(source);

	if !path.is_file() {
		return Ok(None);
	}

	let text = std::fs::read_to_string(&path)?;
	let read = import(&text).map_err(|error| err!(Asset("{}: {error}", path.display())))?;

	Ok(Some(read))
}

/// Reads a sidecar.
///
/// @param text - the whole file
pub fn import(text: &str) -> Result<Import> {
	let root = crate::json::parse(text)?;
	let table = names(Transform::FIELDS, &HAND);

	check(&root, &[table], &HAND, "an import")?;

	let mut transform = Transform::IDENTITY;

	read(&mut transform, &root, Transform::FIELDS, "an import")?;

	if transform.matrix().determinant() < 0.0 {
		return Err(err!(Asset(
			"an import that mirrors turns every triangle in the file inside out, and colby \
			 culls one way round; scale by a positive number and turn it instead"
		)));
	}

	Ok(Import {
		transform,
		skip: skipped(&root)?,
		materials: remapped(&root)?,
		levels: counted(&root)?,
	})
}

/// Writes a sidecar.
///
/// @param sidecar - what to write
/// @return the text, or why it could not be written
pub fn export(sidecar: &Import) -> Result<String> {
	let mut rows = Rows::default();

	put_all(
		&mut rows,
		&sidecar.transform,
		&Transform::IDENTITY,
		Transform::FIELDS,
		&Writing {
			prefix: "",
			what: "an import",
			skipped: &[],
		},
		|_| None,
	)?;

	if !sidecar.skip.is_empty() {
		let names: Vec<String> = sidecar
			.skip
			.iter()
			.map(|name| as_text(name))
			.collect();

		rows.put("skip", format!("[{}]", names.join(", ")));
	}

	if !sidecar.materials.is_empty() {
		let pairs: Vec<String> = sidecar
			.materials
			.iter()
			.map(|(from, to)| format!("{}: {}", as_text(from), as_text(to)))
			.collect();

		rows.put("materials", format!("{{ {} }}", pairs.join(", ")));
	}

	if let Some(levels) = sidecar.levels {
		rows.put("levels", levels.to_string());
	}

	Ok(rows.text())
}

/// Refuses what a sidecar beside a `.obj` cannot mean.
///
/// A `.cmesh` is one mesh with no name of its own and no materials in it, so
/// two of the three keys have nothing to act on. Refused rather than ignored,
/// for the reason a `.cscene` refuses a body kind it has no word for: half of
/// a file being obeyed silently is worse than the whole of it being read.
///
/// @param sidecar - what the file said
pub fn check_mesh_only(sidecar: &Import) -> Result<()> {
	if !sidecar.skip.is_empty() {
		return Err(err!(Asset(
			"an OBJ compiles to one mesh with no names in it, so there is nothing for skip to \
			 leave out"
		)));
	}

	if !sidecar.materials.is_empty() {
		return Err(err!(Asset(
			"an OBJ carries no materials colby reads, so there is nothing for materials to \
			 remap; write the material on the entity instead"
		)));
	}

	Ok(())
}

/// How far a composed placement may drift before it is worth a word.
#[must_use]
pub const fn is_sheared(drift: f32) -> bool { drift > SQUARE_ENOUGH }

/// The node names to leave out.
fn skipped(root: &Value) -> Result<Vec<String>> {
	let Some(value) = root.get("skip") else {
		return Ok(Vec::new());
	};

	let Value::Array(items) = value else {
		return Err(err!(Asset("an import's skip is a list of the node names to leave out")));
	};

	items
		.iter()
		.map(|item| {
			item.as_str()
				.map(str::to_owned)
				.ok_or_else(|| err!(Asset("an import skips a node by name, not by number")))
		})
		.collect()
}

/// The most coarser levels a mesh is compiled with, when the sidecar says.
///
/// A whole number up to [`MAX_LEVELS`], refused otherwise rather than clamped:
/// a count of twelve is somebody expecting twelve, and a mesh carrying seven is
/// not what they were told.
fn counted(root: &Value) -> Result<Option<usize>> {
	let Some(value) = root.get("levels") else {
		return Ok(None);
	};

	value
		.as_usize()
		.filter(|levels| *levels <= MAX_LEVELS)
		.map(Some)
		.ok_or_else(|| {
			err!(Asset(
				"an import's levels is how many coarser levels a mesh is compiled with, a whole \
				 number from 0 to {MAX_LEVELS}"
			))
		})
}

/// What each of the file's materials is really made of.
fn remapped(root: &Value) -> Result<Vec<(String, String)>> {
	let Some(value) = root.get("materials") else {
		return Ok(Vec::new());
	};

	if !matches!(value, Value::Object(_)) {
		return Err(err!(Asset(
			"an import's materials maps the name the file gave a material to the asset name of \
			 one to wear instead"
		)));
	}

	value
		.as_object()
		.iter()
		.map(|(from, to)| {
			let name = to.as_str().ok_or_else(|| {
				err!(Asset("an import's {from} should be the asset name of a material"))
			})?;

			Ok((from.clone(), name.to_owned()))
		})
		.collect()
}

#[cfg(test)]
mod tests {
	use colby_core::glam::{Quat, Vec3};

	use super::*;

	/// A sidecar with all three of its answers given.
	fn sample() -> Import {
		Import {
			transform: Transform {
				position: Vec3::new(0.0, 1.0, 0.0),
				rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
				scale: Vec3::splat(0.01),
			},
			skip: vec!["collision_proxy".to_owned(), "helper".to_owned()],
			materials: vec![("brass".to_owned(), "materials/brass".to_owned())],
			levels: Some(3),
		}
	}

	#[test]
	fn a_sidecar_is_named_after_the_whole_source_file() {
		assert_eq!(
			beside(Path::new("assets/models/lamp.gltf")),
			PathBuf::from("assets/models/lamp.gltf.model"),
			"the whole name plus the suffix, so two sources in one directory cannot share one"
		);
		assert_eq!(
			beside(Path::new("assets/models/lamp.obj")),
			PathBuf::from("assets/models/lamp.obj.model"),
			"and the OBJ beside it has its own"
		);
	}

	#[test]
	fn everything_written_comes_back() {
		let text = export(&sample()).expect("it can be written");
		let read = import(&text).unwrap_or_else(|failure| {
			panic!("what was written did not read back: {failure}\n{text}")
		});

		assert_eq!(read.skip, sample().skip, "the list, in order: {text}");
		assert_eq!(read.materials, sample().materials, "and the map: {text}");
		assert_eq!(read.levels, Some(3), "and the count: {text}");
		assert!(
			read.transform
				.position
				.abs_diff_eq(sample().transform.position, 1e-6),
			"and where it lands: {text}"
		);
		assert!(
			read.transform
				.scale
				.abs_diff_eq(sample().transform.scale, 1e-6),
			"scale and all: {text}"
		);
	}

	#[test]
	fn a_file_that_says_nothing_is_silent_and_changes_nothing() {
		let read = import("{}").expect("an empty object is a sidecar");

		assert!(read.is_silent(), "it says nothing");
		assert_eq!(read, Import::NONE);
		assert_eq!(
			read.applied(Transform::at(Vec3::X)).0,
			Transform::at(Vec3::X),
			"and a placement through it is the placement"
		);
	}

	#[test]
	fn a_field_nobody_declared_is_refused_by_name() {
		let refused = import(r#"{ "lods": 4 }"#).expect_err("there are no LODs here");

		assert!(
			format!("{refused}").contains("lods"),
			"and the message says which word it was: {refused}"
		);
	}

	#[test]
	fn the_transform_is_spelled_the_way_a_scene_spells_one() {
		// the whole reason it is a `Transform` and not three fields of its
		// own: one described form, and the inspector draws it for nothing.
		let read = import(r#"{ "position": [1, 2, 3], "scale": [2, 2, 2] }"#)
			.expect("a scene's own words");

		assert_eq!(read.transform.position, Vec3::new(1.0, 2.0, 3.0));
		assert_eq!(read.transform.scale, Vec3::splat(2.0));

		// and every key is the field's own name, so a field added to the
		// table tomorrow is a key here today
		let text = export(&sample()).expect("it can be written");

		for field in Transform::FIELDS {
			assert!(
				text.contains(&format!("\"{}\"", field.name)),
				"{} is in the text: {text}",
				field.name
			);
		}
	}

	#[test]
	fn a_skip_that_is_not_a_list_of_names_is_refused() {
		assert!(import(r#"{ "skip": "helper" }"#).is_err(), "one name is still a list");
		assert!(import(r#"{ "skip": [7] }"#).is_err(), "a node is named, not numbered");
		assert_eq!(
			import(r#"{ "skip": ["helper"] }"#)
				.expect("a list of one")
				.skip,
			vec!["helper".to_owned()]
		);
	}

	#[test]
	fn a_mesh_is_compiled_with_every_level_it_thins_out_to_unless_the_sidecar_says_fewer() {
		assert_eq!(Import::NONE.levels(), MAX_LEVELS, "no sidecar is as many as there are");
		assert_eq!(
			import("{}").expect("an empty sidecar").levels(),
			MAX_LEVELS,
			"and so is one that does not say"
		);
		assert_eq!(
			import(r#"{ "levels": 0 }"#)
				.expect("none at all")
				.levels(),
			0,
			"a mesh only ever drawn whole"
		);
		assert_eq!(
			import(r#"{ "levels": 2 }"#)
				.expect("two")
				.levels(),
			2,
			"and two is two"
		);
		assert_eq!(
			import(&format!(r#"{{ "levels": {MAX_LEVELS} }}"#))
				.expect("as many as a mesh may carry")
				.levels(),
			MAX_LEVELS,
			"the most there may be"
		);
		assert!(
			!import(r#"{ "levels": 0 }"#)
				.expect("none")
				.is_silent(),
			"and it says something"
		);
	}

	#[test]
	fn a_count_of_levels_that_is_not_one_a_mesh_can_have_is_refused() {
		for wrong in ["8", "2.5", "-1", r#""four""#, "[2]", "true"] {
			let refused = import(&format!(r#"{{ "levels": {wrong} }}"#))
				.expect_err("not a count of levels");

			assert!(
				format!("{refused}").contains("levels"),
				"{wrong} is refused, naming the key: {refused}"
			);
		}
	}

	#[test]
	fn a_remap_that_is_not_a_map_of_names_is_refused() {
		assert!(import(r#"{ "materials": ["brass"] }"#).is_err(), "a map, not a list");
		assert!(import(r#"{ "materials": { "brass": 7 } }"#).is_err(), "to a name");
	}

	#[test]
	fn a_node_is_taken_unless_it_is_skipped() {
		let sidecar = sample();

		assert!(!sidecar.takes("collision_proxy"), "what it names is left out");
		assert!(sidecar.takes("shade"), "and everything else is taken");
		assert!(Import::NONE.takes("collision_proxy"), "a silent one takes everything");
	}

	#[test]
	fn a_material_is_remapped_only_where_the_file_says_so() {
		let sidecar = sample();

		assert_eq!(sidecar.material("brass"), Some("materials/brass"));
		assert_eq!(sidecar.material("stone"), None, "the rest stay the model's own");
	}

	#[test]
	fn an_import_that_mirrors_is_refused_rather_than_drawn_inside_out() {
		let refused = import(r#"{ "scale": [-1, 1, 1] }"#).expect_err("a mirror");

		assert!(
			format!("{refused}").contains("inside out"),
			"and the message says what would have happened: {refused}"
		);
		assert!(import(r#"{ "scale": [1, 1, 1] }"#).is_ok(), "and a positive one is fine");
	}

	#[test]
	fn a_placement_lands_where_the_import_puts_it() {
		let sidecar = Import {
			transform: Transform {
				position: Vec3::Y,
				rotation: Quat::IDENTITY,
				scale: Vec3::splat(2.0),
			},
			..Import::NONE
		};
		let (landed, drift) = sidecar.applied(Transform::at(Vec3::X));

		assert!(
			landed
				.position
				.abs_diff_eq(Vec3::new(2.0, 1.0, 0.0), 1e-5),
			"scaled by the import and then moved by it: {landed:?}"
		);
		assert!(landed.scale.abs_diff_eq(Vec3::splat(2.0), 1e-5));
		assert!(!is_sheared(drift), "a uniform scale shears nothing: {drift}");
	}

	#[test]
	fn an_uneven_scale_over_a_turned_placement_is_called_a_shear() {
		let sidecar = Import {
			transform: Transform {
				scale: Vec3::new(1.0, 4.0, 1.0),
				..Transform::IDENTITY
			},
			..Import::NONE
		};
		let placed = Transform {
			rotation: Quat::from_rotation_z(std::f32::consts::FRAC_PI_4),
			..Transform::IDENTITY
		};

		assert!(
			is_sheared(sidecar.applied(placed).1),
			"the same arithmetic the flattening warns about, and the same answer"
		);
	}

	#[test]
	fn a_sidecar_beside_an_obj_may_only_move_it() {
		assert!(check_mesh_only(&Import::NONE).is_ok(), "silence is fine");
		assert!(
			check_mesh_only(&Import {
				transform: sample().transform,
				..Import::NONE
			})
			.is_ok(),
			"and so is a transform, which is the one an OBJ most wants"
		);
		assert!(
			check_mesh_only(&Import { levels: Some(0), ..Import::NONE }).is_ok(),
			"and a count of levels, which one mesh has as much as any"
		);

		let refused = check_mesh_only(&sample()).expect_err("skip has nothing to act on");

		assert!(format!("{refused}").contains("skip"), "and it says which: {refused}");
	}

	#[test]
	fn there_is_no_sidecar_where_there_is_no_file() {
		let dir = std::env::temp_dir().join("colby-import-none");

		drop(std::fs::create_dir_all(&dir));

		let source = dir.join("lamp.gltf");

		drop(std::fs::write(&source, "{}"));
		drop(std::fs::remove_file(beside(&source)));

		assert_eq!(
			read_beside(&source).expect("no file is not a failure"),
			None,
			"having none is the ordinary case"
		);

		drop(std::fs::write(beside(&source), r#"{ "scale": [2, 2, 2] }"#));

		let found = read_beside(&source)
			.expect("a file that reads")
			.expect("is a sidecar");

		assert_eq!(found.transform.scale, Vec3::splat(2.0));
		assert!(!found.is_silent(), "and it says something");

		drop(std::fs::remove_dir_all(&dir));
	}
}
