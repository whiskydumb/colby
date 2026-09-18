//! Working a world's still light out into a picture, from the console, the
//! editor and the command line.
//!
//! Three ways in and one function behind them: `light.bake <name>` in a
//! window, the editor's button, which types it, and `--bake [scene]` with no
//! window, no device and no game module at all. Each bakes a world, keeps the
//! picture as a source at `assets/lightmaps/<name>.hdr`, writes where each
//! thing's light is on the picture into that thing's `baking` record, names
//! the picture on the world, and **writes the scene out as
//! `assets/scenes/<name>.scene` with it**.
//!
//! **The scene is written with the picture, always, and that is not a
//! convenience.** Where each thing's light is lives in the scene - a record on
//! each entity - and the picture lives in its own file; a picture written
//! without its scene would be read, the next time the scene is loaded, through
//! the places of the bake before it. So the two are one write, and a bake is
//! refused while the world is playing, because what a game has done to a world
//! is not what anybody means to save.
//!
//! **A bake stops the frame it runs in**, as long as it takes, and the log says
//! how long each part took. A bake in the background would have to know, when
//! it came back, that the world it writes into is still the world it baked -
//! that no scene was loaded meanwhile - which is a mechanism of its own; an
//! editor that stops for a bake is what the field's full bakers do.
//!
//! **Three settings, saved like the navmesh's**, because how finely a
//! project's light is baked belongs to the project: [`TEXELS`], [`BOUNCES`] and
//! [`RAYS`]. There is a command where the navmesh has none because this bake
//! is not cheap enough to run every time something changes, and because it
//! writes files.

use std::{fs, path::Path, time::Instant};

use colby_asset::{Project, compile::LIGHTMAPS, radiance};
use colby_bake::{Baked, Placeless, Rect, Scene, Settings};
use colby_core::{
	Result,
	abi::{BAKING, EntityId, World, cvar::Value, scene},
	err, error, info, warn,
};

use crate::{
	Asked,
	assets::Assets,
	console,
	saves::{self, SOURCES},
};

/// `light.bake <name>`: works the world's still light out, and writes the
/// picture and the scene.
pub(crate) const BAKE: &str = "light.bake";

/// How many texels a unit of surface a bake gives, at the least.
pub(crate) const TEXELS: &str = "bake.texels";

/// How many gathers a bake makes.
pub(crate) const BOUNCES: &str = "bake.bounces";

/// How many rays a texel sends in each.
pub(crate) const RAYS: &str = "bake.rays";

/// Registers the command and the three settings.
///
/// @param world - the table to register into
pub(crate) fn install(world: &mut World) {
	world.cvars.command(
		BAKE,
		colby_core::abi::console::defer,
		"work out the still light of the world into assets/lightmaps/<name>.hdr, and write the \
		 scene as assets/scenes/<name>.scene with it",
	);
	world.cvars.saved(
		TEXELS,
		Value::Float(Settings::DEFAULT.texels),
		"how many texels a unit of surface a bake gives, at the least",
	);
	world.cvars.saved(
		BOUNCES,
		Value::Int(i64::from(Settings::DEFAULT.bounces)),
		"how many gathers a bake makes: a lamp's light off that many surfaces, the sky's off \
		 one fewer",
	);
	world.cvars.saved(
		RAYS,
		Value::Int(i64::from(Settings::DEFAULT.rays)),
		"how many rays each texel sends in each gather of a bake",
	);
}

/// What a bake is asked for, as the table says it.
///
/// @param world - where the three settings live
#[must_use]
pub(crate) fn settings(world: &World) -> Settings {
	let whole = |name: &str, fallback: u32| {
		world
			.cvars
			.int(name)
			.and_then(|value| u32::try_from(value.max(0)).ok())
			.unwrap_or(fallback)
	};

	Settings {
		texels: world
			.cvars
			.float(TEXELS)
			.unwrap_or(Settings::DEFAULT.texels),
		bounces: whole(BOUNCES, Settings::DEFAULT.bounces),
		rays: whole(RAYS, Settings::DEFAULT.rays),
	}
	.sane()
}

/// Bakes, if a line asked for it.
///
/// The frame loop's, once a frame and outside a step, like every command that
/// writes a file. The last line wins, as with a scene's.
///
/// @param world - the world to bake
/// @param project - whose asset tree to write into
pub(crate) fn serve(world: &mut World, project: &Project) {
	let Some(asked) = console::take(world, &[BAKE]).pop() else {
		return;
	};

	if !crate::mode::wanted(world) {
		error!(
			"a bake writes the scene it bakes, and a world that is playing is not the scene; \
			 stop the world first"
		);

		return;
	}

	if let Err(failure) = bake(world, project, &asked.words.join(" "), settings(world)) {
		error!(%failure, "the bake could not be made");
	}
}

/// `--bake [scene]`: bakes a scene with no window, no device and no game.
///
/// The asset tree compiled and loaded, the scene put in place exactly as it
/// was written, baked with the project's settings and whatever `--set` said,
/// and both files written; the process then ends. No game module, because what
/// a game's code adds to a world when it starts is not the scene, and writing
/// it into the scene would put it there twice.
///
/// @param project - the project
/// @param scene - the scene's name under `assets/scenes/`, or the one the
/// project starts as
/// @param asked - what `--set` said
///
/// # Errors
///
/// If there is no such scene, if it holds nothing to bake, or if a file cannot
/// be written.
pub(crate) fn run(project: &Project, scene: Option<&str>, asked: &Asked) -> Result {
	let name = match scene {
		| Some(name) => name.to_owned(),
		| None => project
			.startup_scene()
			.and_then(|name| name.strip_prefix(&format!("{SOURCES}/")))
			.map(str::to_owned)
			.ok_or_else(|| {
				err!(Asset(
					"--bake names no scene and the project starts as none under {SOURCES}/; \
					 name one"
				))
			})?,
	};
	let mut world = World::new();

	Assets::of(project).sync(&mut world);

	let id = world.scenes.find(&format!("{SOURCES}/{name}"));

	if !id.is_some() {
		return Err(err!(Asset(
			"there is no scene {SOURCES}/{name}; it compiles from assets/{SOURCES}/{name}.scene"
		)));
	}

	let data = world.scenes.data(id).clone();

	scene::restore(&mut world, &data)?;
	bake(&mut world, project, &name, asked_settings(asked, &project.settings()))
}

/// What a bake is asked for before there is a table: the command line, then
/// the project's settings file, by name - the order the table would apply
/// them in, read the way the graphics API is read.
///
/// @param asked - what `--set` said
/// @param archive - the project's settings file
fn asked_settings(asked: &Asked, archive: &Path) -> Settings {
	let said = |name: &str| {
		asked
			.last(name)
			.map(str::to_owned)
			.or_else(|| console::archived(archive, name))
	};
	let whole = |name: &str, fallback: u32| {
		said(name)
			.and_then(|value| value.parse().ok())
			.unwrap_or(fallback)
	};

	Settings {
		texels: said(TEXELS)
			.and_then(|value| value.parse().ok())
			.unwrap_or(Settings::DEFAULT.texels),
		bounces: whole(BOUNCES, Settings::DEFAULT.bounces),
		rays: whole(RAYS, Settings::DEFAULT.rays),
	}
	.sane()
}

/// Bakes a world, and writes the picture and the scene.
///
/// @param world - the world, which gets each thing's place and the picture's
/// name
/// @param project - whose asset tree
/// @param name - what the picture and the scene are called
/// @param settings - what is asked for
///
/// # Errors
///
/// If the name is not a plain file name, if nothing can be baked, or if a file
/// cannot be written.
fn bake(world: &mut World, project: &Project, name: &str, settings: Settings) -> Result {
	let name = saves::plain(name)?;
	let started = Instant::now();
	let scene = Scene::of(world);

	info!(
		name,
		texels = settings.texels,
		bounces = settings.bounces,
		rays = settings.rays,
		threads = colby_bake::threads(),
		"a bake starts"
	);

	let baked = colby_bake::bake(&scene, settings, colby_bake::threads())?;

	report(world, &baked);

	let picture = kept(world, project, name, &baked)?;

	placed(world, &baked);

	let source = saves::source(&project.assets(), name)?;
	let bytes = saves::written(world, &source)?;

	info!(
		picture = %picture.display(),
		scene = %source.display(),
		bytes,
		seconds = started.elapsed().as_secs_f64(),
		"baked"
	);

	Ok(())
}

/// Writes the picture as a source, and puts what the compiler will make of it
/// in the registry at once.
///
/// At once rather than when the compiler next looks, because the scene written
/// a moment later names the picture by what it is called in the registry, and
/// a picture the registry does not hold yet has no name there. The same bytes
/// the compiler will read, through the same function, so the texture it loads
/// a quarter of a second later is this one and nothing is uploaded twice.
///
/// @return where the picture was written
fn kept(
	world: &mut World,
	project: &Project,
	name: &str,
	baked: &Baked,
) -> Result<std::path::PathBuf> {
	let texels: Vec<[f32; 3]> = baked
		.light
		.iter()
		.map(colby_core::glam::Vec3::to_array)
		.collect();
	let bytes = radiance::encode(baked.width, baked.height, &texels)?;
	let path = project
		.assets()
		.join(LIGHTMAPS)
		.join(name)
		.with_extension(radiance::EXTENSION);

	if let Some(directory) = path.parent() {
		fs::create_dir_all(directory)?;
	}

	fs::write(&path, &bytes)?;

	let data = radiance::lightmap(&radiance::import(&bytes)?)?;

	world.lightmap = world
		.textures
		.insert(&format!("{LIGHTMAPS}/{name}"), data);

	info!(
		path = %path.display(),
		width = baked.width,
		height = baked.height,
		bytes = bytes.len(),
		"lightmap written"
	);

	Ok(path)
}

/// Writes every thing's place on the picture into its record, and nought into
/// every other thing's, so that nothing keeps a place from a bake before; the
/// flag that leaves a thing out is left as it was.
fn placed(world: &mut World, baked: &Baked) {
	let ids: Vec<EntityId> = world.entities.iter().map(|(id, ..)| id).collect();

	for id in ids {
		let place = baked
			.places
			.iter()
			.find(|(entity, _)| *entity == id)
			.map_or_else(Rect::default, |(_, place)| *place);
		let Some(baking) = world.entities.record_mut(&BAKING, id) else {
			continue;
		};
		let [left, top, width, height] = [place.left, place.top, place.width, place.height]
			.map(|side| i32::try_from(side).unwrap_or(i32::MAX));

		baking.left = left;
		baking.top = top;
		baking.width = width;
		baking.height = height;
	}
}

/// What the bake did, into the log: each part and how long it took, the
/// counts, and every mesh that has no place on the picture and why.
fn report(world: &World, baked: &Baked) {
	let report = &baked.report;

	for (part, took) in &report.parts {
		info!(part = part.as_str(), ms = took.as_secs_f64() * 1000.0, "a part of the bake");
	}

	info!(
		pieces = report.pieces,
		triangles = report.triangles,
		lamps = report.lamps,
		samples = report.samples,
		ring = report.ring,
		pushed = report.pushed,
		buried = report.buried,
		rays = report.rays,
		width = baked.width,
		height = baked.height,
		"what was baked"
	);

	let mut named: Vec<(String, Placeless)> = baked
		.placeless
		.iter()
		.map(|(_, mesh, why)| {
			let name = world.meshes.get(*mesh).map_or_else(
				|| "a mesh with no name".to_owned(),
				|entry| entry.name().to_owned(),
			);

			(name, *why)
		})
		.collect();

	named.sort_by(|one, other| one.0.cmp(&other.0));
	named.dedup_by(|one, other| one.0 == other.0);

	for (mesh, why) in named {
		match why {
			| Placeless::Unlit => {},
			| Placeless::NoSheet => warn!(
				mesh,
				"a still mesh has no second set of coordinates, so it keeps no baked light; its \
				 sidecar may have turned the unwrap off"
			),
			| Placeless::Oversized => warn!(
				mesh,
				"a still mesh's sheet is wider than a lightmap can be, so it keeps no baked \
				 light"
			),
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{env, path::PathBuf};

	use colby_asset::{AlignedBytes, TextureFile, compile, scene as file};
	use colby_core::{
		abi::{MeshId, Renderable, Transform, material::MaterialId},
		glam::Vec3,
	};

	use super::*;
	use crate::Launch;

	/// A workspace nothing else is using.
	fn workspace(name: &str) -> PathBuf {
		let inside = env::temp_dir().join(format!("colby_test_light_{name}"));
		drop(fs::remove_dir_all(&inside));

		inside
	}

	/// A project under that workspace, starting as the scene given.
	fn project(name: &str, startup: Option<&str>) -> Project {
		let text = startup.map_or_else(
			|| r#"{ "schema": 1, "engine": "0.1.0", "id": "testing", "name": "a test" }"#.to_owned(),
			|scene| {
				format!(
					r#"{{ "schema": 1, "engine": "0.1.0", "id": "testing", "name": "a test", "startup_scene": "{scene}" }}"#
				)
			},
		);

		Project::parse(&workspace(name), &text).expect("a project")
	}

	/// A floor and a block standing on it, under a sky, with the console's
	/// variables and the mode registered, the world being edited.
	fn room() -> World {
		let mut world = World::new();

		console::install(&mut world);
		world.cvars.set(crate::mode::EDIT, "1");
		world.editing = true;
		world.light = Vec3::new(-0.3, -1.0, 0.2);

		for (mesh, at, scale) in [
			(MeshId::QUAD, Vec3::ZERO, Vec3::new(6.0, 1.0, 6.0)),
			(MeshId::CUBE, Vec3::new(0.5, 0.5, 0.0), Vec3::ONE),
		] {
			let thing = world.entities.spawn_at(Transform {
				position: at,
				scale,
				..Transform::IDENTITY
			});

			world
				.entities
				.set_renderable(thing, Renderable::of(mesh, MaterialId::DEFAULT, Vec3::ONE));
		}

		world.cvars.set(RAYS, "16");
		world.cvars.set(BOUNCES, "1");

		world
	}

	#[test]
	fn a_bake_writes_the_picture_the_scene_and_every_thing_s_place() {
		let project = project("writes", None);
		let mut world = room();
		let stale = world.entities.spawn();

		// a place from a bake before, on a thing with no mesh, which a bake
		// takes away
		if let Some(baking) = world.entities.record_mut(&BAKING, stale) {
			baking.width = 9;
			baking.height = 9;
		}

		colby_core::abi::console::run(&mut world, "light.bake yard");
		serve(&mut world, &project);

		let picture = project
			.assets()
			.join("lightmaps")
			.join("yard.hdr");
		let written = project.assets().join("scenes").join("yard.scene");
		let read = radiance::import(&fs::read(&picture).expect("the picture is there"))
			.expect("and it is a picture");

		assert!(written.is_file(), "the scene is written with it");
		assert!(world.lightmap.is_some(), "the world names the picture");
		assert_eq!(
			world
				.textures
				.get(world.lightmap)
				.map(|entry| entry.name().to_owned()),
			Some("lightmaps/yard".to_owned()),
			"by the name the compiler gives it"
		);

		let places: Vec<(EntityId, [i32; 4])> = world
			.entities
			.iter()
			.filter_map(|(id, ..)| {
				let baking = world.entities.record(&BAKING, id)?;

				Some((id, [baking.left, baking.top, baking.width, baking.height]))
			})
			.collect();

		for (id, [left, top, width, height]) in &places {
			if *id == stale {
				assert_eq!([*width, *height], [0, 0], "a place from a bake before is taken away");

				continue;
			}

			assert!(*width > 0 && *height > 0, "{id:?} has a place");
			assert!(
				u32::try_from(left + width).expect("a place") <= read.width
					&& u32::try_from(top + height).expect("a place") <= read.height,
				"inside the picture of {}x{}",
				read.width,
				read.height
			);
		}

		// and the scene says all of it, read back as the compiler will
		let text = fs::read_to_string(&written).expect("the scene reads");
		let scene = colby_asset::level::import(&text).expect("and it is a scene");

		assert_eq!(scene.lightmap, "lightmaps/yard", "the picture's name is in the scene");
		assert!(text.contains("\"baking\""), "and each thing's place is in its record: {text}");
	}

	#[test]
	fn the_picture_compiles_to_the_texture_the_bake_put_in_the_registry() {
		let project = project("compiles", None);
		let mut world = room();

		colby_core::abi::console::run(&mut world, "light.bake yard");
		serve(&mut world, &project);

		let report = compile::compile_dir(&project.assets(), &project.output(), false)
			.expect("the tree compiles");

		assert!(report.failed.is_empty(), "nothing failed: {:?}", report.failed);

		let compiled = report
			.compiled
			.iter()
			.find(|compiled| compiled.name == "lightmaps/yard")
			.expect("the picture compiled");
		let file = TextureFile::open(&compiled.output).expect("a texture");
		let registered = world
			.textures
			.get(world.lightmap)
			.map(|entry| entry.value().clone())
			.expect("the registry holds it");

		assert_eq!(file.to_texture_data(), registered, "the same texels the bake put there");

		let scene = report
			.compiled
			.iter()
			.find(|compiled| compiled.name == "scenes/yard")
			.expect("the scene compiled");
		let read = file::SceneFile::from_bytes(AlignedBytes::read(&scene.output).expect("bytes"))
			.expect("a scene")
			.to_scene_data();

		assert_eq!(read.lightmap, "lightmaps/yard", "and the compiled scene names it");
	}

	#[test]
	fn a_bake_is_refused_while_the_world_plays() {
		let project = project("plays", None);
		let mut world = room();

		world.cvars.set(crate::mode::EDIT, "0");
		colby_core::abi::console::run(&mut world, "light.bake yard");
		serve(&mut world, &project);

		assert!(!project.assets().join("lightmaps").exists(), "no picture");
		assert!(!project.assets().join("scenes").exists(), "and no scene");
		assert!(world.asked.is_empty(), "and the line is taken, not left for later");
	}

	#[test]
	fn a_windowless_bake_bakes_the_scene_the_project_starts_as() {
		let project = project("windowless", Some("scenes/yard"));
		let mut world = room();

		// the scene written the way an editor writes it, then baked from the
		// command line with no window and no game
		saves::written(&world, &project.assets().join("scenes").join("yard.scene"))
			.expect("the scene is written");
		world.cvars.set(RAYS, "16");

		run(&project, None, &Asked::default()).expect("it bakes");

		let text = fs::read_to_string(project.assets().join("scenes").join("yard.scene"))
			.expect("the scene is there");

		assert!(
			project
				.assets()
				.join("lightmaps")
				.join("yard.hdr")
				.is_file(),
			"the picture"
		);
		assert!(text.contains("\"lightmap\": \"lightmaps/yard\""), "named in the scene: {text}");

		assert!(
			run(&project, Some("nowhere"), &Asked::default()).is_err(),
			"a scene nobody wrote is refused by name"
		);
	}

	#[test]
	fn a_windowless_bake_reads_its_settings_from_the_line_then_the_project() {
		let project = project("settings", None);
		let archive = project.settings();

		fs::create_dir_all(archive.parent().expect("a directory")).expect("made");
		fs::write(&archive, "bake.rays 64\nbake.bounces 2\n").expect("written");

		let line = Launch::parse(&["--set".to_owned(), "bake.rays=32".to_owned()]).asked;
		let asked = asked_settings(&line, &archive);

		assert_eq!(asked.rays, 32, "the line first");
		assert_eq!(asked.bounces, 2, "then the project");
		assert_eq!(
			asked.texels.to_bits(),
			Settings::DEFAULT.texels.to_bits(),
			"then what a bake starts with"
		);
	}
}
