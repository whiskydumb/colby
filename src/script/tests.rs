//! What a document's script may and may not do, checked without a window.
//!
//! Everything here builds a `World` by hand, registers a program and a document
//! naming it, and drives [`Vm::update`] the way the step does. No GPU, no files
//! and no interpreter state left over between tests: each one gets its own
//! [`Vm`], which is also the only way to be sure two documents are isolated
//! because they are given separate environments rather than because nothing has
//! collided yet.

use colby_core::abi::{
	Body, BodyId, EntityId, PeerId, ScriptData, Shape, Transform, Value, World,
	ui::{DocumentData, Event, EventKind, PanelId},
};

use super::{Asked, Vm};

/// A world showing one document, with this program registered under its name.
///
/// Two registrations rather than one, because a program is an asset now: the
/// document holds the name and the table holds the text, exactly as the
/// compiler and the runner leave them.
fn showing(script: &str) -> (World, PanelId) {
	let mut world = World::new();
	world
		.scripts
		.insert("ui/test", ScriptData { source: script.to_owned() });
	world.ui.insert("ui/test", document("ui/test"));
	let panel = world.ui.show("ui/test");

	(world, panel)
}

/// Somebody editing the program's own file, which is what the compiler does to
/// the table.
fn rewritten(world: &mut World, script: &str) {
	world
		.scripts
		.insert("ui/test", ScriptData { source: script.to_owned() });
}

/// A world with one program under the world's own directory and no panel at
/// all.
fn running(script: &str) -> World {
	let mut world = World::new();
	world
		.scripts
		.insert("scripts/test", ScriptData { source: script.to_owned() });

	world
}

/// What a program has left in a console variable, which is the only thing a
/// world program can write to today.
fn said(world: &World) -> Option<String> { Some(world.cvars.get("script.said")?.value()?.text()) }

/// Registers the variable a world program writes into, so a test can read
/// something back out of one.
fn listen(world: &mut World) {
	world
		.cvars
		.var("script.said", Value::Text(String::new()), "what a test program wrote");
}

/// A document that names one program and holds nothing else.
fn document(program: &str) -> DocumentData {
	DocumentData {
		program: program.to_owned(),
		..DocumentData::empty()
	}
}

/// What a test registers a published command with.
///
/// A **safe** extern function, which coerces to a `ConsoleFn` without a word of
/// `unsafe` anywhere: this crate has none in it by design, and a stand-in that
/// dereferences nothing needs none. What the runner installs instead writes
/// down what was asked; here nothing types anything, and the tests hand the
/// interpreter a list directly.
extern "C-unwind" fn nothing(_world: *mut World, _args: *const colby_core::abi::Args) {}

/// An interpreter with that stand-in in it.
fn machine() -> Vm { Vm::new(nothing).expect("the interpreter starts") }

/// One whole step's worth of the interpreter, which is two calls.
///
/// The step drives the two halves at two different moments - the interface
/// before the physics and the world's own after the game's `update` - and
/// almost every test here is about what one step does, so this is the fixture
/// rather than either half on its own.
fn stepped(scripts: &mut Vm, world: &mut World) {
	scripts.interface(world);
	scripts.gameplay(world, &[]);
}

/// Puts one event in the queue, the way the interface does.
fn happened(world: &mut World, panel: PanelId, node: &str, kind: EventKind) {
	world
		.ui
		.push_event(Event { panel, node: node.to_owned(), kind });
}

/// What the game - or the script - has written into a node.
fn written(world: &World, panel: PanelId, node: &str) -> Option<String> {
	world.ui.panel(panel)?.bind(node)?.text.clone()
}

/// The classes it has been given.
fn classes(world: &World, panel: PanelId, node: &str) -> Option<String> {
	world.ui.panel(panel)?.bind(node)?.classes.clone()
}

#[test]
fn a_document_with_no_script_is_not_a_failure() {
	let (mut world, panel) = showing("");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "hold", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(scripts.loaded.len(), 1, "the panel is known, so it is not looked at again");
	assert!(scripts.loaded[0].handlers.is_none(), "with nothing to hand an event to");
}

#[test]
fn a_click_reaches_the_handler_that_asked_for_it() {
	let (mut world, panel) =
		showing(r#"ui.on("hold", "click", function() ui.set_text("hold", "holding") end)"#);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	assert_eq!(written(&world, panel, "hold"), None, "nothing has happened yet");

	happened(&mut world, panel, "hold", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "hold").as_deref(),
		Some("holding"),
		"the click ran the handler and the handler wrote through the panel"
	);
}

#[test]
fn a_handler_can_read_a_field_back_and_not_only_write_to_it() {
	// the one read in the whole surface, and a search box is why: a handler
	// that cannot see what somebody typed has nothing to search for.
	let (mut world, panel) = showing(
		r#"ui.on("hold", "click", function() ui.set_text("hold", ui.text("hold") .. "!") end)"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	world.ui.set_text(panel, "hold", "typed");

	happened(&mut world, panel, "hold", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "hold").as_deref(),
		Some("typed!"),
		"the handler read what was there and wrote it back with a mark on it"
	);
}

#[test]
fn a_handler_answers_only_the_kind_it_registered() {
	let (mut world, panel) =
		showing(r#"ui.on("hold", "click", function() ui.set_text("hold", "clicked") end)"#);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "hold", EventKind::Press);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "hold"),
		None,
		"a press is not a click, and the script said click"
	);
}

#[test]
fn the_handler_is_told_which_node_and_which_kind() {
	let (mut world, panel) = showing(
		r#"ui.on("a", "press", function(node, kind) ui.set_text("out", node .. ":" .. kind) end)"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "a", EventKind::Press);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("a:press"),
		"so that one function can serve several boxes"
	);
}

#[test]
fn editing_the_program_replaces_it() {
	let (mut world, panel) =
		showing(r#"ui.on("hold", "click", function() ui.set_text("hold", "first") end)"#);
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	// what the compiler does when the file is edited: the same name, a new
	// value, and an entry whose revision has moved.
	rewritten(
		&mut world,
		r#"ui.on("hold", "click", function() ui.set_text("hold", "second") end)"#,
	);

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "hold", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "hold").as_deref(),
		Some("second"),
		"the new program answers, and the old one is gone rather than stacked under it"
	);
}

#[test]
fn rewriting_the_document_does_not_restart_a_program_that_did_not_change() {
	// what a program being an asset actually buys, and the case that produces
	// it constantly: a stylesheet is folded into the document, so editing
	// `theme.css` rewrites `hud.cdoc` and moves its revision while the program
	// beside it has not been touched. Watching the document's revision would
	// throw away every local in the program every time somebody changed a
	// color.
	// the counter is kept in a *bind* rather than in a local, and that is the
	// whole of what makes this test able to fail: a rebuild hands the chunk a
	// fresh environment, so a program counting its own loads in a global reads
	// `1` whether it ran once or five times. A bind is the host's and outlives
	// the program that wrote it.
	let (mut world, panel) =
		showing(r#"ui.set_text("out", tostring((tonumber(ui.text("out")) or 0) + 1))"#);
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	// the same name, a different document, a revision that moved.
	let mut restyled = document("ui/test");
	restyled
		.rules
		.push(colby_core::abi::ui::Rule::default());
	world.ui.insert("ui/test", restyled);

	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("1"),
		"the program was not run again, so its locals are where it left them"
	);
}

#[test]
fn a_document_that_names_something_else_runs_something_else() {
	// the other half of the rule above: nothing watches the document's
	// revision, so the thing that has to notice a document being repointed is
	// the handle it resolves to.
	let (mut world, panel) = showing(r#"ui.set_text("out", "the first")"#);
	world.scripts.insert("ui/other", ScriptData {
		source: r#"ui.set_text("out", "the second")"#.to_owned(),
	});
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	world.ui.insert("ui/test", document("ui/other"));
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("the second"),
		"the panel runs what its document now names"
	);
}

#[test]
fn a_program_that_arrives_after_the_document_is_picked_up() {
	// the case a revision alone cannot see: a fresh registry entry starts at
	// revision zero exactly as the null entry does, so a document naming a
	// program nobody had compiled yet resolves to nothing at revision nought,
	// and the program landing afterwards resolves to something at revision
	// nought. Only the handle moved.
	let mut world = World::new();
	world.ui.insert("ui/test", document("ui/late"));
	let panel = world.ui.show("ui/test");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	assert!(written(&world, panel, "out").is_none(), "there was nothing to run");

	world.scripts.insert("ui/late", ScriptData {
		source: r#"ui.set_text("out", "here")"#.to_owned(),
	});
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("here"),
		"and the program compiled after the document was loaded still runs"
	);
}

#[test]
fn what_a_script_wrote_outlives_the_document_it_came_from() {
	let (mut world, panel) =
		showing(r#"ui.on("score", "click", function() ui.set_text("score", "1200") end)"#);
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "score", EventKind::Click);
	stepped(&mut scripts, &mut world);

	rewritten(&mut world, "-- nothing at all");
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "score").as_deref(),
		Some("1200"),
		"a bind is the host's, and reloading a document replaces its content rather than what \
		 was laid over it"
	);
}

#[test]
fn a_program_that_does_not_compile_leaves_the_last_one_running() {
	let (mut world, panel) =
		showing(r#"ui.on("hold", "click", function() ui.set_text("hold", "working") end)"#);
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	rewritten(&mut world, "function (");
	stepped(&mut scripts, &mut world);

	happened(&mut world, panel, "hold", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "hold").as_deref(),
		Some("working"),
		"the same answer a shader that will not compile gives: keep the last one that did"
	);
}

#[test]
fn a_broken_program_is_reported_once_rather_than_every_step() {
	let (mut world, _) = showing("function (");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	let after = scripts.loaded[0].program_revision;
	stepped(&mut scripts, &mut world);

	assert_eq!(scripts.loaded.len(), 1, "one panel, one entry");
	assert_eq!(
		scripts.loaded[0].program_revision, after,
		"the revision is recorded even when the load failed, so nothing retries sixty times a \
		 second; the next edit moves it again"
	);
}

#[test]
fn a_handler_that_will_not_stop_is_stopped() {
	let (mut world, panel) = showing(
		r#"ui.on("a", "click", function()
			ui.set_text("a", "started")
			while true do end
		end)"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "a", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "a").as_deref(),
		Some("started"),
		"it ran, it was cut off at the instruction budget, and the step went on"
	);
}

#[test]
fn a_handler_that_asks_for_more_memory_than_it_may_have_is_stopped() {
	// one call, so the instruction budget never sees it: this is the case the
	// memory limit exists for and the reason it is not the same knob twice.
	let (mut world, panel) = showing(
		r#"ui.on("a", "click", function()
			ui.set_text("a", "started")
			local hungry = string.rep("x", 100 * 1024 * 1024)
			ui.set_text("a", tostring(#hungry))
		end)"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "a", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "a").as_deref(),
		Some("started"),
		"a hundred megabytes in one instruction is refused, the handler stops there, and the \
		 step goes on"
	);
}

#[test]
fn a_script_cannot_reach_the_filesystem_or_the_clock() {
	let (mut world, panel) = showing(
		r#"ui.set_classes("out", tostring(os) .. " " .. tostring(io) .. " " ..
			tostring(require) .. " " .. tostring(load) .. " " .. tostring(dofile))"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		classes(&world, panel, "out").as_deref(),
		Some("nil nil nil nil nil"),
		"none of them is in an environment this crate builds by hand"
	);
}

#[test]
fn a_script_has_the_libraries_it_needs_to_be_useful() {
	let (mut world, panel) = showing(
		r#"ui.set_text("out", string.upper("ok") .. " " .. tostring(#{1, 2}) .. " " ..
			tostring(math.floor(1.7)))"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("OK 2 1"),
		"string, table and math are open, which is what an interface script is written with"
	);
}

#[test]
fn two_documents_do_not_share_their_globals() {
	let mut world = World::new();
	world.scripts.insert("ui/first", ScriptData {
		source: "shared = 'from the first'".to_owned(),
	});
	world.scripts.insert("ui/second", ScriptData {
		source: r#"ui.set_text("out", tostring(shared))"#.to_owned(),
	});
	world.ui.insert("ui/first", document("ui/first"));
	world
		.ui
		.insert("ui/second", document("ui/second"));

	let first = world.ui.show("ui/first");
	let second = world.ui.show("ui/second");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert!(first != second, "two documents are two panels");
	assert_eq!(
		written(&world, second, "out").as_deref(),
		Some("nil"),
		"each program gets an environment of its own, so one cannot write into another's"
	);
}

#[test]
fn two_interpreters_roll_the_same_dice() {
	let script = r#"ui.set_text("out", tostring(math.random(1, 1000000)))"#;
	let (mut first, one) = showing(script);
	let (mut second, two) = showing(script);

	machine().interface(&mut first);
	machine().interface(&mut second);

	assert_eq!(
		written(&first, one, "out"),
		written(&second, two, "out"),
		"Lua 5.4 seeds a fresh state from the clock and an address, and a document that rolled \
		 a die would then make two runs of `--shot` different pictures"
	);
}

#[test]
fn a_handler_can_ask_the_console_for_something() {
	let (mut world, panel) =
		showing(r#"ui.on("go", "click", function() colby.command("test.value 7") end)"#);
	world
		.cvars
		.var("test.value", Value::Int(0), "a number for a test to move");

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "go", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		world.cvars.int("test.value"),
		Some(7),
		"the one bridge out of the interface, and it goes through the table the game itself \
		 declared into"
	);
}

#[test]
fn an_event_nobody_has_is_refused_when_it_is_registered() {
	let (mut world, panel) = showing(
		r#"ui.on("a", "hover", function() ui.set_text("a", "never") end)
		   ui.on("a", "click", function() ui.set_text("a", "also never") end)"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	happened(&mut world, panel, "a", EventKind::Click);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "a"),
		None,
		"a misspelled event stops the chunk where it stands rather than registering nothing and \
		 looking like a handler that never fires"
	);
}

#[test]
fn a_style_written_from_a_script_is_read_by_the_same_parser_the_document_uses() {
	let (mut world, panel) = showing(r#"ui.set_style("fill", "width: 37%")"#);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let width = world
		.ui
		.panel(panel)
		.and_then(|panel| panel.bind("fill"))
		.and_then(|bind| bind.style.width);

	assert_eq!(
		width,
		Some(colby_core::abi::ui::Length::Percent(37.0)),
		"which is the one thing a class cannot say"
	);
}

#[test]
fn a_script_is_only_run_when_something_it_was_built_from_has_moved() {
	// as above: counting in a bind rather than in a local, because a fresh
	// environment per load makes a global counter say `1` forever.
	let (mut world, panel) =
		showing(r#"ui.set_text("out", tostring((tonumber(ui.text("out")) or 0) + 1))"#);
	let mut scripts = machine();

	for _ in 0..8 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("1"),
		"eight steps, one load: nothing runs a chunk again until the compiler rewrites it"
	);
	assert!(panel.is_some(), "and the panel resolved in the first place");
}

#[test]
fn a_program_under_the_world_directory_runs_with_nobody_showing_it() {
	// nothing loads it: the host walks the table for the prefix. There is no
	// panel here at all, which is the point - a world program is not a
	// document's.
	let mut world = running(r#"colby.command("script.said hello")"#);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("hello"), "the chunk ran");
	assert_eq!(scripts.loaded.len(), 1, "and one program is being kept track of");
}

#[test]
fn a_program_outside_that_directory_is_not_the_worlds_to_run() {
	// the whole of the rule. `ui/test` is a program the table holds and
	// nothing shows, so nothing runs it: a document has to name it.
	let mut world = World::new();
	listen(&mut world);
	world.scripts.insert("ui/test", ScriptData {
		source: r#"colby.command("script.said anyway")"#.to_owned(),
	});

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some(""), "nobody ran it");
	assert!(scripts.loaded.is_empty(), "and there is nothing to keep track of");
}

#[test]
fn a_world_program_has_no_panel_to_write_to() {
	// not refused, and not an error: the environment is built by hand, so a
	// name nobody declared is `nil` - which is what Lua answers for anything
	// that is not there. There is no branch anywhere saying no.
	let mut world = running(r#"colby.command("script.said " .. type(ui))"#);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("nil"), "there is no `ui` in the environment");
}

#[test]
fn editing_a_world_program_runs_it_again() {
	let mut world = running(r#"colby.command("script.said first")"#);
	listen(&mut world);
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	world.scripts.insert("scripts/test", ScriptData {
		source: r#"colby.command("script.said second")"#.to_owned(),
	});
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("second"), "the new one ran");
	assert_eq!(scripts.loaded.len(), 1, "and replaced the old rather than standing beside it");
}

#[test]
fn a_world_program_is_run_once_and_not_every_step() {
	// the counter is in a console variable rather than in a local, because a
	// rebuild hands the chunk a fresh environment and a global counter would
	// read `1` however many times it ran. Same fixture fault the panel tests
	// had.
	let mut world = running(
		r#"local n = tonumber(colby.command) -- deliberately not a number
		colby.command("script.said " .. (n or "once"))"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	for _ in 0..8 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(said(&world).as_deref(), Some("once"), "eight steps, and it ran");
	assert_eq!(scripts.loaded.len(), 1, "once");
}

#[test]
fn touching_a_program_is_what_makes_it_run_again() {
	// `script.reload` in one line: nothing is read off disk and the source is
	// the same one, and the revision moving is the whole signal.
	let mut world = running(r#"colby.command("script.said " .. tostring(colby.command ~= nil))"#);
	listen(&mut world);
	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	world.cvars.set("script.said", "cleared");
	stepped(&mut scripts, &mut world);
	assert_eq!(said(&world).as_deref(), Some("cleared"), "nothing moved, so nothing ran");

	let id = world.scripts.find("scripts/test");
	assert!(world.scripts.touch(id), "and now it is asked for again");
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "so the same source ran a second time");
}

#[test]
fn two_programs_do_not_share_the_tables_the_engine_gives_them() {
	// `Table` is a reference, so handing the same one to every program makes
	// `colby.command = nil` in one of them everybody's problem until the next
	// step rebuilds the table. Each program gets a projection instead: reads
	// fall through, writes land in front.
	//
	// The names are chosen so the vandal is walked first - the table is
	// walked in slot order and both are world programs.
	let mut world = World::new();
	listen(&mut world);
	world.scripts.insert("scripts/first", ScriptData {
		source: "colby.command = function() end".to_owned(),
	});
	world
		.scripts
		.insert("scripts/second", ScriptData {
			source: r#"colby.command("script.said through")"#.to_owned(),
		});

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("through"),
		"the second program reached the engine's own function rather than the first's"
	);
}

#[test]
fn a_world_program_that_will_not_compile_is_reported_and_left_alone() {
	let mut world = running("function (");
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	stepped(&mut scripts, &mut world);

	assert_eq!(scripts.loaded.len(), 1, "one program, one entry");
	assert!(scripts.loaded[0].handlers.is_none(), "and nothing was built out of it");
}

#[test]
fn a_world_program_files_no_handlers_because_it_has_nowhere_to_file_one() {
	// `ui.on` is not a name a world program can reach, so an empty table
	// standing in for its handlers would be a claim the log then repeats.
	let mut world = running("-- a program that does nothing at all");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(scripts.loaded.len(), 1, "it ran");
	assert!(scripts.loaded[0].handlers.is_none(), "and filed nothing");
}

#[test]
fn a_panel_and_the_world_are_two_programs_even_over_one_name() {
	// the two halves are keyed differently on purpose: a panel by the panel,
	// because two panels showing one document are two programs with separate
	// locals, and the world by the asset, because there is one of each.
	let mut world = World::new();
	listen(&mut world);
	world.scripts.insert("scripts/both", ScriptData {
		source: r#"colby.command("script.said ran")"#.to_owned(),
	});
	world
		.ui
		.insert("ui/test", document("scripts/both"));
	let panel = world.ui.show("ui/test");

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert!(panel.is_some(), "the panel resolved");
	assert_eq!(scripts.loaded.len(), 2, "one program, two homes: the panel's and the world's");
}

#[test]
fn a_world_program_is_called_once_a_step() {
	// the door step thirteen deliberately kept shut, and the whole of this
	// commit. The counter is in a console variable rather than in a local
	// because a rebuild would hand the chunk a fresh environment - but here
	// that is also the assertion, since a program that were rebuilt each step
	// would count one every time.
	let mut world = running(
		r#"local ran = 0
		function tick(dt)
			ran = ran + 1
			colby.command("script.said " .. ran)
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	for _ in 0..5 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(said(&world).as_deref(), Some("5"), "five steps, five calls, one program");
}

#[test]
fn a_tick_is_handed_the_length_of_a_step() {
	let mut world = running(r#"function tick(dt) colby.command("script.said " .. dt) end"#);
	listen(&mut world);
	world.dt = 0.25;

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("0.25"), "which is what the step is on");
}

#[test]
fn a_program_that_wants_no_tick_is_not_called_and_costs_nothing() {
	let mut world = running("-- a program that runs once and is done");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(scripts.loaded.len(), 1, "it ran");
	assert!(!scripts.loaded[0].ticks, "and asked for nothing further");
	assert!(!scripts.anything_ticks(), "so no step opens a scope for it");
}

#[test]
fn a_program_can_stop_itself_by_forgetting_its_own_tick() {
	// what reading the tick out of the environment every step buys, against
	// resolving it once at load. A program that has finished says so.
	let mut world = running(
		r#"function tick(dt)
			colby.command("script.said once")
			tick = nil
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	world.cvars.set("script.said", "cleared");

	for _ in 0..4 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(said(&world).as_deref(), Some("cleared"), "it was not called again");
}

#[test]
fn a_document_that_asks_for_a_tick_is_told_where_a_tick_belongs() {
	// not refused - the rest of the program is fine and its handlers work -
	// but not called either, and said out loud rather than left as a function
	// nobody explains.
	let (mut world, panel) = showing(r#"function tick(dt) ui.set_text("out", "ticked") end"#);
	let mut scripts = machine();

	for _ in 0..4 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(scripts.loaded.len(), 1, "the panel's program loaded");
	assert!(!scripts.loaded[0].ticks, "and is not ticked");
	assert!(written(&world, panel, "out").is_none(), "so it never ran");
}

#[test]
fn a_tick_that_will_not_stop_is_stopped_and_switched_off() {
	// the one failure worth muting: an ordinary error costs almost nothing and
	// a runaway costs the whole budget every step for as long as nobody looks.
	let mut world = running(
		r#"function tick(dt)
			colby.command("script.said started")
			while true do end
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("started"), "it got as far as the loop");
	assert!(scripts.loaded[0].muted, "and was switched off");
	assert_eq!(scripts.loaded[0].faults, 1, "with the failure counted");
	assert!(
		scripts.loaded[0].instructions > super::BUDGET,
		"and the count is what tells a budget apart from an ordinary error: {}",
		scripts.loaded[0].instructions
	);

	world.cvars.set("script.said", "cleared");
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("cleared"), "and is not called again");
}

#[test]
fn a_muted_program_starts_again_when_it_is_asked_for() {
	let mut world = running(
		r#"function tick(dt)
			colby.command("script.said ran")
			while true do end
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	assert!(scripts.loaded[0].muted, "switched off");

	// `script.reload` in one line: the revision moves, the entry is built
	// again, and a fresh entry is never muted.
	let id = world.scripts.find("scripts/test");
	assert!(world.scripts.touch(id));
	world.cvars.set("script.said", "cleared");
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("ran"), "it was asked for and it ran");
}

#[test]
fn a_tick_that_fails_is_counted_and_the_program_goes_on() {
	// the other half of the rule: an ordinary error is not worth switching a
	// program off for, because the next step may be fine.
	let mut world = running(
		r#"local n = 0
		function tick(dt)
			n = n + 1
			colby.command("script.said " .. n)
			error("something went wrong")
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	for _ in 0..3 {
		stepped(&mut scripts, &mut world);
	}

	assert!(!scripts.loaded[0].muted, "an ordinary error does not switch it off");
	assert_eq!(scripts.loaded[0].faults, 3, "and every one of them is counted");
	assert_eq!(said(&world).as_deref(), Some("3"), "and it ran three times");
}

#[test]
fn nothing_is_loaded_or_ticked_while_the_world_is_being_edited() {
	// what the two entry points are for. `gameplay` is the half the step keeps
	// inside its edit-mode guard, so a test that only drives `interface` is
	// what a step in edit mode does.
	//
	// **There is a panel here, and it is doing something, which is the whole
	// reason this test can fail.** `interface` returns before it opens a scope
	// at all when no panel is pending and nothing was clicked, so a world with
	// no interface in it cannot tell a tick in the wrong half from no tick:
	// the early return would hide it. A person editing a world still clicks
	// the buttons on it, and that is the case this drives.
	let (mut world, panel) = showing(r#"ui.on("go", "click", function() end)"#);
	listen(&mut world);
	world.scripts.insert("scripts/test", ScriptData {
		source: r#"function tick(dt) colby.command("script.said ran") end"#.to_owned(),
	});
	let mut scripts = machine();

	for _ in 0..4 {
		happened(&mut world, panel, "go", EventKind::Click);
		scripts.interface(&mut world);
		world.ui.end_step();
	}

	assert_eq!(said(&world).as_deref(), Some(""), "nothing of the world's ran at all");
	assert_eq!(scripts.loaded.len(), 1, "and only the panel's program was loaded");

	stepped(&mut scripts, &mut world);
	assert_eq!(said(&world).as_deref(), Some("ran"), "it starts the moment play resumes");

	// and the half a cold start cannot see: with the program *already* loaded,
	// editing has to stop it ticking rather than merely stop it loading.
	world.cvars.set("script.said", "cleared");

	for _ in 0..4 {
		happened(&mut world, panel, "go", EventKind::Click);
		scripts.interface(&mut world);
		world.ui.end_step();
	}

	assert_eq!(said(&world).as_deref(), Some("cleared"), "and stops again while editing");
}

#[test]
fn two_world_programs_are_ticked_in_the_order_the_table_holds_them() {
	// slot order, which is the order the compiler walked a sorted tree in.
	// Nothing here may depend on a hash, because a screenshot runs these.
	let mut world = World::new();
	listen(&mut world);
	world.scripts.insert("scripts/a", ScriptData {
		source: r#"function tick(dt) colby.command("script.said a") end"#.to_owned(),
	});
	world.scripts.insert("scripts/b", ScriptData {
		source: r#"function tick(dt) colby.command("script.said b") end"#.to_owned(),
	});

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("b"), "the second one wrote last");
	assert_eq!(
		scripts
			.loaded
			.iter()
			.map(|one| one.name.as_str())
			.collect::<Vec<&str>>(),
		["scripts/a", "scripts/b"],
		"in slot order"
	);
}

#[test]
fn one_muted_program_does_not_stop_the_others() {
	// the check inside the loop, which the early return alone cannot stand in
	// for: with a second program still ticking there *is* work to do this
	// step, so the loop is entered and every muted entry in it has to be
	// stepped over one at a time.
	let mut world = World::new();
	listen(&mut world);
	world.scripts.insert("scripts/a", ScriptData {
		source: r#"function tick(dt)
			colby.command("script.said runaway")
			while true do end
		end"#
			.to_owned(),
	});
	world.scripts.insert("scripts/b", ScriptData {
		source: r#"function tick(dt) colby.command("script.said healthy") end"#.to_owned(),
	});

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert!(scripts.loaded[0].muted, "the first one was switched off");
	assert!(!scripts.loaded[1].muted, "and the second was not");

	// the second still has work, so the step does not return early and the
	// loop really is walked over the muted one.
	for _ in 0..3 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(said(&world).as_deref(), Some("healthy"), "only the healthy one wrote");
	assert_eq!(scripts.loaded[0].faults, 1, "and the muted one was never called again");
}

/// A world with one named entity, one named body and a program over them.
///
/// The two are given the *same* slot and generation on purpose: both tables are
/// dense and both start at slot nought, so this is the arrangement a real world
/// is in, and it is the one an untagged handle cannot tell apart.
fn peopled(script: &str) -> (World, EntityId, BodyId) {
	let mut world = running(script);
	listen(&mut world);

	let entity = world.entities.spawn();
	world.entities.set_name(entity, "crate");

	let body = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0).driving(entity));
	world.bodies.set_name(body, "crate body");

	(world, entity, body)
}

#[test]
fn a_program_finds_a_thing_by_the_name_it_was_given() {
	// **the comparison is done in Lua and one word comes back.** A console
	// variable is set from the second word of the line, so a value with a
	// space in it arrives truncated and an assertion over it is an assertion
	// about its first word. @ref `console::dispatch`.
	let mut world = World::new();
	listen(&mut world);

	let entity = world.entities.spawn();
	world.entities.set_name(entity, "crate");
	let body = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0).driving(entity));
	world.bodies.set_name(body, "crate");

	// the same slot and the same generation in two tables, which is the
	// arrangement a real world is always in and the one an untagged handle
	// cannot tell apart.
	assert_eq!(entity.slot(), body.slot(), "the same slot");
	assert_eq!(entity.generation(), body.generation(), "and the same generation");

	world.scripts.insert("scripts/test", ScriptData {
		source: format!(
			r#"function tick(dt)
				local one = colby.describe(entity.find("crate")) == "entity {slot}:{age}"
				local two = colby.describe(body.find("crate")) == "body {slot}:{age}"
				colby.command("script.said " .. tostring(one) .. tostring(two))
			end"#,
			slot = entity.slot(),
			age = entity.generation()
		),
	});

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("truetrue"),
		"one name, two tables, two different handles"
	);
}

#[test]
fn a_name_nothing_answers_to_is_nothing_rather_than_a_handle_to_nothing() {
	let (mut world, ..) = peopled(
		r#"function tick(dt)
			colby.command("script.said " .. tostring(entity.find("nobody")))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("nil"),
		"which is what `if entity.find(..) then` reads, and a handle to nothing is not"
	);
}

#[test]
fn a_handle_survives_being_kept_from_one_step_to_the_next() {
	// the whole commit in one test: a program keeps a handle in a local, in a
	// table, and as a table *key*, and all three still name the same thing
	// four steps later. A value that had to be made fresh each step could do
	// none of the three.
	let (mut world, ..) = peopled(
		r#"local mine = {}
		local kept = nil
		local step = 0

		function tick(dt)
			step = step + 1

			if step == 1 then
				kept = entity.find("crate")
				mine[kept] = "the crate"
			end

			if step == 4 then
				colby.command("script.said " .. tostring(mine[kept] ~= nil) .. tostring(entity.valid(kept)))
			end
		end"#,
	);
	let mut scripts = machine();

	for _ in 0..4 {
		stepped(&mut scripts, &mut world);
	}

	assert_eq!(
		said(&world).as_deref(),
		Some("truetrue"),
		"it is still the key it was, and still names something alive"
	);
}

#[test]
fn a_handle_to_something_that_has_gone_says_so() {
	let (mut world, entity, _) = peopled(
		r#"local kept = nil

		function tick(dt)
			kept = kept or entity.find("crate")
			colby.command("script.said " .. tostring(entity.valid(kept)))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	assert_eq!(said(&world).as_deref(), Some("true"), "alive to start with");

	world.entities.despawn(entity);
	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("false"),
		"and the generation is what says so, rather than the slot being empty"
	);
}

#[test]
fn a_slot_taken_over_by_somebody_else_is_not_the_thing_that_was_there() {
	// the reason a handle carries a generation at all, and the failure it
	// exists to stop: the next spawn takes the slot that was just freed.
	let (mut world, entity, _) = peopled(
		r#"local kept = nil

		function tick(dt)
			kept = kept or entity.find("crate")
			colby.command("script.said " .. tostring(entity.valid(kept)))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	world.entities.despawn(entity);

	let taken = world.entities.spawn();
	assert_eq!(taken.slot(), entity.slot(), "the same slot, which is the trap");
	assert_ne!(taken, entity, "and a different handle");

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("false"), "so the old handle names nothing");
}

#[test]
fn a_handle_from_the_other_table_is_refused_by_name() {
	// what the tag buys. Untagged this would be a confident answer about a
	// different thing, because both tables are dense and both start at slot
	// nought generation one.
	//
	// The message is searched inside Lua and one word comes back, for the
	// reason the test above says: a console variable keeps the second word of
	// the line and nothing after it.
	let (mut world, ..) = peopled(
		r#"function tick(dt)
			local ok, why = pcall(function() return body.name(entity.find("crate")) end)
			local named = tostring(why):find("entity") ~= nil
			colby.command("script.said " .. tostring(ok) .. tostring(named))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("falsetrue"),
		"it was refused, and told which table it was really holding"
	);
}

#[test]
fn a_number_that_is_not_a_handle_is_refused_rather_than_read_as_a_slot() {
	let (mut world, ..) = peopled(
		r#"function tick(dt)
			local ok = pcall(function() return entity.name(1) end)
			colby.command("script.said " .. tostring(ok))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("false"), "one is a number, not slot one");
}

#[test]
fn nothing_is_not_an_error_because_finding_nothing_is_ordinary() {
	let (mut world, ..) = peopled(
		r#"function tick(dt)
			colby.command("script.said " .. tostring(body.valid(body.find("nope"))))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("false"), "the chain everybody writes first");
}

#[test]
fn everything_in_a_table_is_handed_over_in_slot_order() {
	// slot order rather than a Lua hash's, because `--shot` runs these and two
	// runs have to be one picture. @ref `colby-known-gaps` on `pairs`.
	let (mut world, ..) = peopled(
		r#"function tick(dt)
			local all = entity.all()
			local names = ""
			for i = 1, #all do
				names = names .. tostring(entity.name(all[i]))
			end
			colby.command("script.said " .. #all .. names)
		end"#,
	);

	for name in ["second", "third"] {
		let extra = world.entities.spawn();
		world.entities.set_name(extra, name);
	}

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("3cratesecondthird"),
		"three of them, in the order the table holds them"
	);
}

#[test]
fn a_panel_has_no_world_and_a_world_program_has_no_panel() {
	// the enforcement, and it is the environment rather than a check: what is
	// not in a program's environment is `nil`, and `nil.find` is a Lua error
	// nobody had to write.
	let (mut world, panel) =
		showing(r#"ui.set_text("out", type(entity) .. type(body) .. type(ui))"#);
	listen(&mut world);
	world.scripts.insert("scripts/test", ScriptData {
		source: r#"colby.command("script.said " .. type(entity) .. type(body) .. type(ui))"#
			.to_owned(),
	});

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("nilniltable"),
		"a panel's program has the interface and no world"
	);
	assert_eq!(
		said(&world).as_deref(),
		Some("tabletablenil"),
		"and a world program has the world and no interface"
	);
}

/// Makes this world one that has joined somebody else's.
///
/// A slot and a generation that are not the host's, which is all it takes:
/// `PeerId::HOST` is slot nought at a generation no allocator ever reaches, so
/// anything else is somebody who was named by a host.
fn joined(world: &mut World) { world.peer = PeerId::from_bits((1_u64 << 32) | 1); }

#[test]
fn a_program_puts_something_in_the_world_and_finds_it_again() {
	let mut world = running(
		r#"function tick(dt)
			if entity.find("made") then return end

			local made = entity.spawn()
			entity.set_name(made, "made")
			entity.set_position(made, 1.5, 2.5, -3.5)
			entity.set_scale(made, 2, 2, 2)
			entity.draw(made, "cube")
			entity.set_color(made, 0.25, 0.5, 0.75)
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let made = world
		.entities
		.iter()
		.find(|(id, ..)| world.entities.name(*id) == "made")
		.map(|(id, ..)| id)
		.expect("the program spawned one and named it");
	let at = world
		.entities
		.transform(made)
		.expect("it is alive");

	assert!(
		at.position
			.abs_diff_eq(colby_core::glam::Vec3::new(1.5, 2.5, -3.5), 1e-5)
	);
	assert!(
		at.scale
			.abs_diff_eq(colby_core::glam::Vec3::splat(2.0), 1e-5)
	);
	assert!(
		world
			.entities
			.renderable(made)
			.is_some_and(|drawn| drawn
				.color
				.abs_diff_eq(colby_core::glam::Vec3::new(0.25, 0.5, 0.75), 1e-5)),
		"and the color it was given"
	);
}

#[test]
fn a_position_read_back_is_the_one_that_was_written() {
	let mut world = running(
		r#"function tick(dt)
			local made = entity.find("made")
			if not made then
				made = entity.spawn()
				entity.set_name(made, "made")
				entity.set_position(made, 3, 4, 5)
				return
			end

			local x, y, z = entity.position(made)
			colby.command("script.said " .. x .. "," .. y .. "," .. z)
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("3.0,4.0,5.0"), "three values, not a table");
}

#[test]
fn a_position_asked_of_something_that_has_gone_is_three_nothings() {
	// the reason the answer is a tuple of options rather than an option of a
	// tuple: `local x, y, z = ...` has to leave `x` nil, so `if x then` reads
	// the way anybody would write it.
	let mut world = running(
		r#"local kept = nil

		function tick(dt)
			if not kept then
				kept = entity.spawn()
				return
			end

			local x, y, z = entity.position(kept)
			colby.command("script.said " .. tostring(x) .. tostring(y) .. tostring(z))
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	let made = world
		.entities
		.iter()
		.next()
		.map(|(id, ..)| id)
		.expect("the program spawned one");
	world.entities.despawn(made);
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("nilnilnil"), "and not a zero anybody could use");
}

#[test]
fn angles_go_in_and_come_back_out() {
	// a quaternion is four numbers nobody writes by hand, so what crosses is
	// three angles and what is stored is the rotation they make.
	let mut world = running(
		r#"function tick(dt)
			local made = entity.find("turned")
			if not made then
				made = entity.spawn()
				entity.set_name(made, "turned")
				entity.set_angles(made, 0.75, -0.25, 0.5)
				return
			end

			local yaw, pitch, roll = entity.angles(made)
			local near = function(a, b) return a - b < 0.001 and b - a < 0.001 end
			colby.command("script.said " .. tostring(
				near(yaw, 0.75) and near(pitch, -0.25) and near(roll, 0.5)
			))
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "the same three, through a quaternion");
}

#[test]
fn a_client_may_not_make_or_destroy_anything() {
	// nothing on the wire says a thing appeared, so a client that spawned
	// would move every slot after it and start driving the wrong things. The
	// refusal is loud rather than silent, because a silent one is exactly the
	// failure it exists to stop.
	let mut world = running(
		r#"function tick(dt)
			local ok, why = pcall(entity.spawn)
			local named = tostring(why):find("authority") ~= nil
			colby.command("script.said " .. tostring(ok) .. tostring(named))
		end"#,
	);
	listen(&mut world);
	joined(&mut world);

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("falsetrue"), "refused, and told why");
	assert_eq!(world.entities.len(), 0, "and nothing was made");
}

#[test]
fn a_process_on_its_own_is_the_authority_and_is_unaffected() {
	// which is every screenshot, every recording and every window nobody has
	// joined: `World::peer` is the host from `World::new` onwards, and that is
	// a statement rather than a placeholder.
	let mut world = running(
		r#"function tick(dt)
			colby.command("script.said " .. tostring(colby.is_host()))
			entity.spawn()
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"));
	assert_eq!(world.entities.len(), 1, "and the spawn went through");
}

#[test]
fn a_client_may_still_move_what_the_host_made() {
	// the refusal is about *building the table*, not about writing into it: a
	// client drawing a marker over something, or a program deciding a color,
	// changes no slot and is nobody's business but its own.
	let mut world = running(
		r#"function tick(dt)
			local it = entity.find("theirs")
			colby.command("script.said " .. tostring(entity.set_position(it, 9, 9, 9)))
		end"#,
	);
	listen(&mut world);

	let theirs = world.entities.spawn();
	world.entities.set_name(theirs, "theirs");
	joined(&mut world);

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "it was allowed");
	assert!(
		world
			.entities
			.transform(theirs)
			.is_some_and(|at| at.position.x > 8.0),
		"and it moved"
	);
}

#[test]
fn drawing_something_keeps_the_color_it_was_given() {
	// the two are separate concerns - what a thing is drawn as, and what tint
	// is laid over it - so setting one has no business clearing the other.
	// The order here is the one that can fail: color first, then the mesh.
	let mut world = running(
		r#"function tick(dt)
			local made = entity.find("painted")
			if not made then
				made = entity.spawn()
				entity.set_name(made, "painted")
				entity.set_color(made, 0.1, 0.2, 0.3)
				entity.draw(made, "cube")
			end
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let made = world
		.entities
		.iter()
		.find(|(id, ..)| world.entities.name(*id) == "painted")
		.map(|(id, ..)| id)
		.expect("it was made");

	assert!(
		world
			.entities
			.renderable(made)
			.is_some_and(|drawn| drawn
				.color
				.abs_diff_eq(colby_core::glam::Vec3::new(0.1, 0.2, 0.3), 1e-5)),
		"the tint outlived being given a mesh"
	);
}

#[test]
fn a_program_makes_a_body_out_of_a_table_and_it_is_the_one_it_asked_for() {
	let mut world = running(
		r#"function tick(dt)
			if body.find("made") then return end

			local it = entity.spawn()
			entity.set_name(it, "made")

			local made = body.spawn {
				entity = it,
				shape = "ball",
				radius = 0.75,
				mass = 3,
				x = 1, y = 2, z = 3,
				layer = 4,
			}
			body.set_name(made, "made")
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let made = world
		.bodies
		.iter()
		.find(|(id, _)| world.bodies.name(*id) == "made")
		.map(|(id, _)| id)
		.expect("the program made one");
	let body = world.bodies.get(made).expect("it is there");

	assert!((body.shape.radius - 0.75).abs() < 1e-5, "the radius it asked for");
	assert!((body.mass - 3.0).abs() < 1e-5, "and the mass");
	assert!(
		body.transform
			.position
			.abs_diff_eq(colby_core::glam::Vec3::new(1.0, 2.0, 3.0), 1e-5)
	);
	assert_eq!(body.layers.layer, colby_core::abi::Layers::bit(4), "and the layer");
	assert_eq!(body.kind, colby_core::abi::BodyKind::Dynamic, "dynamic, which is the default");
	assert!(
		world.entities.name(body.entity).eq("made"),
		"and it drives the entity it was given"
	);
}

#[test]
fn a_shape_nobody_has_is_refused_by_name() {
	// the one field with no sensible default: a body with no shape is not a
	// body, and guessing one would be a program silently getting something
	// else than it wrote.
	let mut world = running(
		r#"function tick(dt)
			local ok, why = pcall(function() return body.spawn { shape = "cone" } end)
			local named = tostring(why):find("cone") ~= nil
			colby.command("script.said " .. tostring(ok) .. tostring(named))
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("falsetrue"), "refused, naming what was asked for");
	assert_eq!(world.bodies.len(), 0, "and nothing was made");
}

#[test]
fn a_client_may_not_make_a_body_either() {
	let mut world = running(
		r#"function tick(dt)
			local ok = pcall(function() return body.spawn { shape = "ball" } end)
			colby.command("script.said " .. tostring(ok))
		end"#,
	);
	listen(&mut world);
	joined(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("false"));
	assert_eq!(world.bodies.len(), 0, "and the table is the one the host built");
}

#[test]
fn pushing_something_wakes_it_up() {
	// a pile that has settled is asleep, and a speed written onto a sleeping
	// body is a push nothing acts on - which reads as the push not working
	// rather than as the body being asleep.
	let mut world = running(
		r#"function tick(dt)
			local it = body.find("pushed")
			if not it then return end

			body.set_velocity(it, 0, 5, 0)
		end"#,
	);

	let id = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0));
	world.bodies.set_name(id, "pushed");
	if let Some(body) = world.bodies.get_mut(id) {
		body.sleeping = true;
	}

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	let body = world.bodies.get(id).expect("it is there");

	assert!((body.velocity.y - 5.0).abs() < 1e-5, "the push landed");
	assert!(!body.sleeping, "and it is awake to be pushed");
}

#[test]
fn a_teleport_moves_the_entity_too_and_says_it_was_a_jump() {
	// writing the body's transform alone would leave the drawn thing where it
	// was until the next step copied it, and would then draw it sliding across
	// the map. One call does all three.
	let mut world = running(
		r#"function tick(dt)
			local it = body.find("moved")
			if not it then return end

			body.teleport(it, 20, 0, 0)
		end"#,
	);

	let entity = world.entities.spawn();
	let id = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0).driving(entity));
	world.bodies.set_name(id, "moved");

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	// what the step does at the bottom of every one of them, and what a snap
	// is *for*: it writes the past to match the present for whatever asked, so
	// that `lerp(x, x, t)` is `x` for every `t`. Standing in for it here is
	// what makes the second assertion below able to fail.
	world.settle();

	assert!(
		world
			.entities
			.transform(entity)
			.is_some_and(|at| (at.position.x - 20.0).abs() < 1e-5),
		"the drawn thing went with it"
	);
	assert!(
		world
			.entities
			.interpolated(entity, 0.0)
			.is_some_and(|at| (at.position.x - 20.0).abs() < 1e-5),
		"and it is drawn arriving rather than sliding, which is what a snap is"
	);
}

#[test]
fn freezing_is_the_kind_rather_than_a_flag() {
	// in this engine kinematic means gameplay owns the transform and the
	// solver leaves it alone, so a frozen thing needs no flag anywhere and
	// everything piles on it as though it were the map.
	let mut world = running(
		r#"function tick(dt)
			local it = body.find("held")
			if not it then return end

			if body.frozen(it) then
				colby.command("script.said frozen")
			else
				body.freeze(it, true)
			end
		end"#,
	);
	listen(&mut world);

	let id = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0));
	world.bodies.set_name(id, "held");

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		world.bodies.get(id).map(|body| body.kind),
		Some(colby_core::abi::BodyKind::Kinematic),
		"and it is the kind that changed"
	);

	stepped(&mut scripts, &mut world);
	assert_eq!(said(&world).as_deref(), Some("frozen"), "which reads back as frozen");
}

#[test]
fn a_layer_is_how_a_program_asks_which_ones() {
	// the rule the whole sandbox is built on: nothing keeps a list, and
	// "which of them are props" is a walk of the table asking each one.
	let mut world = running(
		r#"function tick(dt)
			local mine = 0
			local all = body.all()
			for i = 1, #all do
				if body.on_layer(all[i], 2) then mine = mine + 1 end
			end
			colby.command("script.said " .. mine)
		end"#,
	);
	listen(&mut world);

	for on in [true, false, true] {
		let id = world
			.bodies
			.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0));

		if let Some(body) = world.bodies.get_mut(id).filter(|_| on) {
			body.layers = colby_core::abi::Layers::single(2);
		}
	}

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("2"), "two of the three answer to it");
}

#[test]
fn a_body_and_the_entity_it_drives_are_reachable_from_each_other() {
	let mut world = running(
		r#"function tick(dt)
			local it = body.find("paired")
			if not it then return end

			local drawn = body.entity(it)
			local back = body.of_entity(drawn)
			colby.command("script.said " .. tostring(back == it))
		end"#,
	);
	listen(&mut world);

	let entity = world.entities.spawn();
	let id = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0).driving(entity));
	world.bodies.set_name(id, "paired");

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "round trip, both ways");
}

#[test]
fn a_program_is_told_what_met_what() {
	let mut world = running(
		r#"function tick(dt)
			local all = body.touches()
			if #all == 0 then return end

			local one = all[1]
			colby.command("script.said " .. tostring(
				body.name(one.first) == "left" and body.name(one.second) == "right"
					and one.began and one.y == 3
			))
		end"#,
	);
	listen(&mut world);

	let left = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0));
	let right = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0));
	world.bodies.set_name(left, "left");
	world.bodies.set_name(right, "right");
	world.bodies.touched(colby_core::abi::Touch {
		first: left,
		second: right,
		kind: colby_core::abi::TouchKind::Began,
		point: colby_core::glam::Vec3::new(1.0, 3.0, 5.0),
		normal: colby_core::glam::Vec3::Y,
	});

	let mut scripts = machine();
	scripts.interface(&mut world);
	scripts.gameplay(&mut world, &[]);

	assert_eq!(said(&world).as_deref(), Some("true"), "both names, the kind and the point");
}

/// A world with the solver's own query table installed, which is what makes a
/// trace answer about anything at all.
///
/// Without it `World::new` installs a stub whose queries report a clean miss -
/// deliberately, so that a unit test or an offscreen capture answers rather
/// than dereferencing a null - and a test over tracing would then be a test
/// over the stub.
fn traced(script: &str) -> (World, Box<colby_physics::Simulation>) {
	let mut world = running(script);
	let simulation = Box::new(colby_physics::Simulation::new());

	world.install_physics(simulation.table());
	listen(&mut world);

	(world, simulation)
}

#[test]
fn a_trace_says_what_it_hit_and_where() {
	let (mut world, mut simulation) = traced(
		r#"function tick(dt)
			local hit = colby.trace_ray(0, 10, 0, 0, -10, 0)
			if not hit then
				colby.command("script.said missed")
				return
			end

			colby.command("script.said " .. tostring(
				body.name(hit.body) == "floor" and hit.y > 0.9 and hit.y < 1.1 and hit.ny > 0.9
			))
		end"#,
	);

	// a wide flat box whose top is at y = 1, so a ray straight down from ten
	// stops at a number a person can write in the assertion.
	let floor = world.bodies.spawn(Body::new(
		colby_core::abi::BodyKind::Static,
		Shape::cuboid(colby_core::glam::Vec3::new(10.0, 1.0, 10.0)),
		Transform::at(colby_core::glam::Vec3::ZERO),
	));
	world.bodies.set_name(floor, "floor");
	simulation.step(&mut world);

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "the name, the height and the normal");
}

#[test]
fn a_trace_that_hits_nothing_is_nothing() {
	let (mut world, _simulation) = traced(
		r#"function tick(dt)
			colby.command("script.said " .. tostring(colby.trace_ray(0, 10, 0, 0, -10, 0)))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("nil"),
		"which is what `if colby.trace_ray(..) then` reads"
	);
}

#[test]
fn a_trace_can_be_told_to_pretend_one_thing_is_not_there() {
	// the case everybody meets first: a trace from a thing hits the thing.
	let (mut world, mut simulation) = traced(
		r#"function tick(dt)
			local me = body.find("me")
			local blind = colby.trace_ray(0, 0, 0, 0, -10, 0)
			local seeing = colby.trace_ray(0, 0, 0, 0, -10, 0, me)

			colby.command("script.said " .. tostring(
				body.name(blind.body) == "me" and body.name(seeing.body) == "floor"
			))
		end"#,
	);

	let me = world.bodies.spawn(Body::new(
		colby_core::abi::BodyKind::Static,
		Shape::ball(1.0),
		Transform::at(colby_core::glam::Vec3::ZERO),
	));
	world.bodies.set_name(me, "me");

	let floor = world.bodies.spawn(Body::new(
		colby_core::abi::BodyKind::Static,
		Shape::cuboid(colby_core::glam::Vec3::new(10.0, 1.0, 10.0)),
		Transform::at(colby_core::glam::Vec3::new(0.0, -6.0, 0.0)),
	));
	world.bodies.set_name(floor, "floor");
	simulation.step(&mut world);

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "one saw itself, the other saw past it");
}

#[test]
fn the_three_axes_are_the_way_a_thing_is_actually_turned() {
	// what gameplay wants out of a rotation is "which way is this facing", and
	// a quarter turn about y is the one case anybody can check by hand:
	// forward, which is negative z at rest, becomes negative x.
	let mut world = running(
		r#"function tick(dt)
			local it = entity.find("turned")
			if not it then return end

			local fx, fy, fz = entity.forward(it)
			local ux, uy, uz = entity.up(it)
			local near = function(a, b) return a - b < 0.001 and b - a < 0.001 end

			colby.command("script.said " .. tostring(
				near(fx, -1) and near(fz, 0) and near(uy, 1)
			))
		end"#,
	);
	listen(&mut world);

	let turned = world.entities.spawn();
	world.entities.set_name(turned, "turned");
	if let Some(at) = world.entities.transform_mut(turned) {
		at.rotation = colby_core::glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
	}

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "a quarter turn, by hand");
}

#[test]
fn the_clock_a_program_reads_is_the_simulations_and_not_the_wall() {
	// there is deliberately no way to ask what time it really is: a program
	// that read a real clock would make two runs of a screenshot two
	// pictures, which is the property the whole interpreter is arranged
	// around.
	let mut world = running(
		r#"function tick(dt)
			colby.command("script.said " .. colby.time() .. "/" .. colby.steps())
		end"#,
	);
	listen(&mut world);
	world.time = 1.25;
	world.steps = 75;

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("1.25/75"), "simulated seconds and steps");
}

#[test]
fn a_program_reads_this_machines_keyboard_by_name() {
	// **the `w` here is held with its edge already gone**, which is what makes
	// this able to fail: a key pressed this instant is both held and pressed,
	// so a build that answered the same question twice would agree with a
	// fixture that only ever asked about one of them.
	let mut world = running(
		r#"function tick(dt)
			colby.command("script.said "
				.. tostring(input.held("w"))
				.. tostring(input.pressed("w"))
				.. tostring(input.pressed("space"))
				.. tostring(input.released("q"))
				.. tostring(input.held("escape")))
		end"#,
	);
	listen(&mut world);

	world.input.set_key(colby_core::abi::Key::W, true);
	world.input.set_key(colby_core::abi::Key::Q, true);
	// what the step does at the bottom of every one of them: the edges
	// describe one step and are cleared, while what is held is not.
	world.input.end_step();
	world
		.input
		.set_key(colby_core::abi::Key::Space, true);
	world
		.input
		.set_key(colby_core::abi::Key::Q, false);

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("truefalsetruetruefalse"),
		"held without its edge, an edge, a release, and one nobody touched"
	);
}

#[test]
fn a_key_nobody_has_is_refused_rather_than_read_as_false() {
	// a name nothing answers to would otherwise be a condition that is
	// quietly never true, which is the worst way for a typo to behave.
	//
	// The name here is a real word that is not a key rather than a misspelling
	// of one: the spell checker reads the strings inside tests too, and a
	// fixture holding a deliberate misspelling fails the gate.
	let mut world = running(
		r#"function tick(dt)
			local ok, why = pcall(function() return input.held("pedal") end)
			local named = tostring(why):find("pedal") ~= nil
			colby.command("script.said " .. tostring(ok) .. tostring(named))
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("falsetrue"), "refused, quoting what was typed");
}

#[test]
fn a_panel_has_no_input_table_either() {
	// the same enforcement as the rest: what a panel's program is given is the
	// interface, and this machine's keyboard is the world's business.
	let (mut world, panel) = showing(r#"ui.set_text("out", type(input) .. type(colby.time))"#);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("nilfunction"),
		"no input, and `colby` is the one table both halves share"
	);
}

/// A world with one sound in it, registered under a name a program can use.
fn audible(script: &str) -> World {
	let mut world = running(script);
	listen(&mut world);
	world
		.sounds
		.insert("sounds/thud", colby_core::abi::SoundData {
			samples: vec![0; 48_000],
			rate: 48_000,
			channels: 1,
		});

	world
}

#[test]
fn a_sound_played_somewhere_is_somewhere() {
	// **checked after the step that plays it and before anything moves it.**
	// Both calls set the place, so a test that played and then moved could
	// not tell a broken one from a broken other: whichever of them was left
	// working would answer for both.
	let mut world = audible(
		r#"function tick(dt)
			if colby.steps() > 1 then return end

			sound.play("sounds/thud", 1, 2, 3, true)
		end"#,
	);
	let mut scripts = machine();
	world.steps = 1;

	stepped(&mut scripts, &mut world);

	let voice = world
		.audio
		.iter()
		.next()
		.map(|(id, _)| id)
		.expect("one voice");
	let playing = world.audio.get(voice).expect("it is there");

	assert!(playing.looping, "it loops, which is what was asked for");
	assert!(playing.positioned, "and it is somewhere rather than simply audible");
	assert!(
		playing
			.at
			.abs_diff_eq(colby_core::glam::Vec3::new(1.0, 2.0, 3.0), 1e-5),
		"where it was played"
	);
}

#[test]
fn a_sound_that_started_flat_and_was_moved_is_somewhere_afterwards() {
	// the other half, and the one that needs a voice which was *not*
	// positioned to begin with: moving one is how a looping sound follows
	// something, and it has to make a flat voice into a placed one.
	let mut world = audible(
		r#"local mine = nil

		function tick(dt)
			if not mine then
				mine = sound.play_flat("sounds/thud", true)
				return
			end

			sound.move(mine, 9, 9, 9)
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let voice = world
		.audio
		.iter()
		.next()
		.map(|(id, _)| id)
		.expect("one voice");

	assert!(
		world
			.audio
			.get(voice)
			.is_some_and(|playing| !playing.positioned),
		"it started simply audible"
	);

	stepped(&mut scripts, &mut world);

	let playing = world.audio.get(voice).expect("it is still going");

	assert!(playing.positioned, "and moving it put it in the world");
	assert!(
		playing
			.at
			.abs_diff_eq(colby_core::glam::Vec3::splat(9.0), 1e-5),
		"where it was moved to"
	);
}

#[test]
fn a_handle_to_a_sound_that_has_ended_turns_nothing_down() {
	// the reason a voice handle is generational: a program that starts a
	// footstep and forgets the handle must not, four seconds later, turn down
	// somebody else's music.
	let mut world = audible(
		r#"local mine = nil

		function tick(dt)
			if not mine then
				mine = sound.play_flat("sounds/thud")
				return
			end

			colby.command("script.said " .. tostring(sound.playing(mine))
				.. tostring(sound.stop(mine)))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let voice = world
		.audio
		.iter()
		.next()
		.map(|(id, _)| id)
		.expect("one voice");
	world.audio.stop(voice);

	// the slot is free, and the next sound takes it - which is the trap.
	let taken = world
		.audio
		.play(colby_core::abi::Voice::flat(world.sounds.find("sounds/thud")));
	assert_eq!(taken.slot(), voice.slot(), "the same slot, which is what makes this a test");

	stepped(&mut scripts, &mut world);

	assert_eq!(
		said(&world).as_deref(),
		Some("falsefalse"),
		"the old handle names nothing and turns nothing off"
	);
	assert!(world.audio.alive(taken), "and the new voice is untouched");
}

#[test]
fn a_sound_nobody_compiled_is_a_quiet_world_rather_than_a_stopped_one() {
	// the rule the whole asset side follows: a name the registry does not
	// answer to is not an error.
	let mut world = audible(
		r#"function tick(dt)
			local it = sound.play_flat("sounds/nothing")
			colby.command("script.said " .. tostring(it ~= nil))
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("true"), "a handle came back all the same");
}

#[test]
fn what_a_program_draws_lasts_one_step() {
	// not a countdown: the table is swept at the *top* of the next step, so a
	// line drawn now survives every frame drawn before then and is gone at the
	// start of the one after. A program that wants it to stay draws it again.
	let mut world = running(
		r#"function tick(dt)
			draw.line(0, 0, 0, 1, 2, 3, 1, 0, 0)
			draw.arrow(0, 0, 0, 0, 1, 0, 0, 1, 0)
			draw.box(1, 1, 1, 0.5, 0.5, 0.5, 0, 0, 1)
			draw.ball(2, 2, 2, 0.5, 1, 1, 0)
			draw.label(3, 3, 3, "here", 1, 1, 1)
		end"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let lines = world.debug.lines().len();
	let labels = world.debug.labels().len();

	assert!(lines > 1, "a line, an arrow, a box and a ball are many segments: {lines}");
	assert_eq!(labels, 1, "and words are the one kind that is not a segment");
	assert!(
		world
			.debug
			.labels()
			.first()
			.is_some_and(|label| label.text == "here"),
		"which says what it was given"
	);

	// what the top of the next step does.
	world.debug.begin_step(1.0);

	assert!(world.debug.is_empty(), "and none of it is there a step later");
}

#[test]
fn a_line_is_drawn_between_the_two_points_it_was_given() {
	let mut world = running("function tick(dt) draw.line(1, 2, 3, 4, 5, 6, 0.5, 0, 0) end");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let line = world
		.debug
		.lines()
		.first()
		.copied()
		.expect("one line");

	assert!(
		line.from
			.abs_diff_eq(colby_core::glam::Vec3::new(1.0, 2.0, 3.0), 1e-5),
		"from where it was told"
	);
	assert!(
		line.to
			.abs_diff_eq(colby_core::glam::Vec3::new(4.0, 5.0, 6.0), 1e-5),
		"to where it was told"
	);
}

#[test]
fn a_panel_can_neither_make_a_noise_nor_draw_over_the_world() {
	let (mut world, panel) =
		showing(r#"ui.set_text("out", type(sound) .. type(draw) .. type(input))"#);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("nilnilnil"),
		"none of the world's three are in a document's environment"
	);
}

/// One command, as the console would hand it over.
fn typed(line: &str) -> Vec<Asked> {
	let mut words = line.split_whitespace().map(str::to_owned);
	let Some(name) = words.next() else {
		return Vec::new();
	};

	vec![Asked { name, words: words.collect() }]
}

#[test]
fn a_program_publishes_a_command_and_is_asked_for_it() {
	// **the discipline the sandbox is already written in**: every action is a
	// named function with a console command in front of it and an optional
	// target. That is what makes gameplay drivable with no mouse in it, and it
	// is the layer the wire already uses for remote calls.
	let mut world = running(
		r#"colby.publish("thruster.on", "switch it on", function(which)
			colby.command("script.said " .. (which or "nobody"))
		end)"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert!(
		world
			.cvars
			.get("thruster.on")
			.is_some_and(colby_core::abi::cvar::Entry::is_command),
		"the console table has it, so `help` lists it and typing it is not a mistake"
	);

	scripts.gameplay(&mut world, &typed("thruster.on crate"));

	assert_eq!(said(&world).as_deref(), Some("crate"), "with the word that followed it");
}

#[test]
fn a_command_a_program_published_is_the_interpreters_rather_than_the_modules() {
	// the one thing that would be quietly wrong: the host sets the console's
	// attribution to the module while one is loaded and leaves it set, so a
	// command registered from inside a step would be dropped by the next
	// gameplay reload.
	let mut world = running(r#"colby.publish("mine.go", "go", function() end)"#);
	world
		.cvars
		.attribute(colby_core::abi::cvar::Owner::Module);

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(
		world
			.cvars
			.get("mine.go")
			.map(colby_core::abi::cvar::Entry::owner),
		Some(colby_core::abi::cvar::Owner::Script),
		"it belongs to the interpreter"
	);

	world.cvars.forget_module();

	assert!(
		world.cvars.get("mine.go").is_some(),
		"so unloading the gameplay module leaves it standing"
	);
}

#[test]
fn a_program_may_not_take_a_name_the_engine_or_the_game_already_has() {
	let mut world = running(
		r#"function tick(dt)
			local ok, why = pcall(function() colby.publish("quit", "no", function() end) end)
			local named = tostring(why):find("already") ~= nil
			colby.command("script.said " .. tostring(ok) .. tostring(named))
		end"#,
	);
	listen(&mut world);
	world
		.cvars
		.command("quit", nothing, "stop the process");

	let mut scripts = machine();
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("falsetrue"), "refused, and said why");
}

#[test]
fn a_document_may_not_publish_anything() {
	// a panel's program is interface logic and its life is its panel's; a
	// console command that went away when somebody hid a window would be a
	// console nobody could rely on.
	let (mut world, panel) = showing(
		r#"local ok = pcall(function() colby.publish("ui.go", "no", function() end) end)
		ui.set_text("out", tostring(ok))"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	assert_eq!(written(&world, panel, "out").as_deref(), Some("false"), "refused");
	assert!(world.cvars.get("ui.go").is_none(), "and nothing was registered");
}

#[test]
fn building_a_program_again_takes_its_old_commands_out_first() {
	// a name the new build no longer registers has to stop answering, or the
	// console table describes a program nobody is running.
	let mut world = running(
		r#"colby.publish("mine.first", "one", function() end)
		colby.publish("mine.second", "two", function() end)"#,
	);
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	assert!(world.cvars.get("mine.first").is_some(), "both are there");
	assert!(world.cvars.get("mine.second").is_some());

	world.scripts.insert("scripts/test", ScriptData {
		source: r#"colby.publish("mine.second", "two", function() end)"#.to_owned(),
	});
	stepped(&mut scripts, &mut world);

	assert!(world.cvars.get("mine.first").is_none(), "the one it dropped is gone");
	assert!(world.cvars.get("mine.second").is_some(), "and the one it kept is not");
}

#[test]
fn a_command_typed_at_a_program_that_has_gone_is_not_an_error() {
	// the line was typed while the program was still there, and the console
	// table is swept a step later. Nothing here should raise.
	let mut world = running("-- a program that publishes nothing");
	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	scripts.gameplay(&mut world, &typed("nobody.go"));

	assert_eq!(scripts.loaded.len(), 1, "and the program is untouched");
	assert_eq!(scripts.loaded[0].faults, 0, "with nothing counted against it");
}

#[test]
fn a_command_is_acted_on_before_the_tick_of_the_step_it_arrives_in() {
	// so that a line typed between two steps is answered by the step that
	// follows it rather than by the one after that.
	let mut world = running(
		r#"local asked = false

		colby.publish("mine.go", "go", function() asked = true end)

		function tick(dt)
			colby.command("script.said " .. tostring(asked))
		end"#,
	);
	listen(&mut world);
	let mut scripts = machine();

	scripts.interface(&mut world);
	scripts.gameplay(&mut world, &typed("mine.go"));

	assert_eq!(
		said(&world).as_deref(),
		Some("true"),
		"the tick of that same step already saw it"
	);
}

/// The sandbox's thruster, as it ships.
///
/// **The file itself rather than a copy of it.** What this is a test of is the
/// thing somebody will read and edit, and a paraphrase in here would go on
/// passing after the real one stopped working. The same bargain the mixer's
/// tests make with the recording that ships beside them.
const THRUSTER: &str = include_str!("../../assets/scripts/thruster.lua");

/// A world with one thruster in it, standing at the origin and pointing up.
fn thrusting() -> (World, BodyId) {
	let mut world = World::new();

	world
		.sounds
		.insert("sounds/hum", colby_core::abi::SoundData {
			samples: vec![0; 48_000],
			rate: 48_000,
			channels: 1,
		});
	world
		.scripts
		.insert("scripts/thruster", ScriptData { source: THRUSTER.to_owned() });

	let drawn = world.entities.spawn();
	let it = world.bodies.spawn(
		Body::dynamic(
			Shape::cuboid(colby_core::glam::Vec3::splat(0.5)),
			Transform::IDENTITY,
			1.0,
		)
		.driving(drawn),
	);
	world.bodies.set_name(it, "thruster");

	// and something else with a body, so that "every thruster" is a claim
	// about the ones that are rather than about the table.
	let other = world
		.bodies
		.spawn(Body::dynamic(Shape::ball(0.5), Transform::IDENTITY, 1.0));
	world.bodies.set_name(other, "crate");

	(world, it)
}

/// Whether nothing at all has been written.
///
/// Exact, and the question really is exact: a thruster nobody has switched on
/// has written no speed at all, so anything but nought is a write nobody asked
/// for and a tolerance would hide the smallest of them.
fn untouched(seen: f32) -> bool { seen == 0.0 }

/// How hard something is being pushed straight up, in newtons.
///
/// The program writes a *force* now rather than a speed, and no solver runs in
/// these tests, so this is where its work lands. What the force then does to a
/// velocity has its own test, with a real solver under it.
fn pushed(world: &World, it: BodyId) -> f32 {
	world
		.bodies
		.get(it)
		.map_or(0.0, |body| body.force.y)
}

/// How fast something is going straight up.
fn rising(world: &World, it: BodyId) -> f32 {
	world
		.bodies
		.get(it)
		.map_or(0.0, |body| body.velocity.y)
}

#[test]
fn the_shipped_thruster_pushes_only_once_it_is_switched_on() {
	let (mut world, it) = thrusting();
	let mut scripts = machine();

	world.dt = 1.0 / 60.0;
	stepped(&mut scripts, &mut world);

	assert!(untouched(pushed(&world, it)), "nothing is pushed until somebody says so");
	assert!(
		world.cvars.get("thruster.on").is_some()
			&& world.cvars.get("thruster.off").is_some()
			&& world.cvars.get("thruster.toggle").is_some(),
		"and the three commands it publishes are in the console table"
	);

	scripts.gameplay(&mut world, &typed("thruster.on"));

	let after = pushed(&world, it);

	assert!(after > 0.0, "and now it pushes: {after}");
	assert!(
		(after - 16.0).abs() < 1e-4,
		"by its own sixteen newtons, and by nothing else: {after}"
	);
	assert!(
		untouched(rising(&world, it)),
		"and it does not write a speed at all any more, which is what makes it obey a mass"
	);
}

#[test]
fn the_shipped_thruster_adds_to_what_is_already_pushing_rather_than_replacing_it() {
	// **a push rather than a set**, and a single tick cannot tell the two
	// apart: after one from nothing they agree exactly. What separates them is
	// a second tick with nothing spending what the first left.
	let (mut world, it) = thrusting();
	let mut scripts = machine();

	world.dt = 1.0 / 60.0;
	// something already moving, to say that a force is not a speed: a program
	// that still wrote a velocity would move this number and this one does not.
	if let Some(body) = world.bodies.get_mut(it) {
		body.velocity = colby_core::glam::Vec3::new(0.0, 3.0, 0.0);
	}

	stepped(&mut scripts, &mut world);
	scripts.gameplay(&mut world, &typed("thruster.on"));

	let once = pushed(&world, it);

	assert!((once - 16.0).abs() < 1e-4, "one tick is one thruster's worth: {once}");

	// no solver runs here, so nothing spends what was accumulated and a second
	// tick lands on top of the first. Under a real step the solver clears it
	// between the two, which is what the test below drives.
	scripts.gameplay(&mut world, &[]);

	let twice = pushed(&world, it);

	assert!(
		(twice - 32.0).abs() < 1e-4,
		"and the second is added to the first rather than replacing it: {twice}"
	);
	assert!(
		(rising(&world, it) - 3.0).abs() < 1e-4,
		"while the speed it was already going is untouched by any of it"
	);
}

#[test]
fn a_program_can_turn_a_body_without_moving_it() {
	// the two calls are one line apart and take the same three numbers, so
	// what tells them apart is which field they land in. A test that only
	// asked whether *something* happened would pass either way round.
	let mut world = running(
		r#"function tick(dt)
			local it = body.find("spun")
			body.spin(it, 0, 5, 0)
		end"#,
	);
	let drawn = world.entities.spawn();
	let spun = world.bodies.spawn(
		Body::dynamic(
			Shape::cuboid(colby_core::glam::Vec3::splat(0.5)),
			Transform::IDENTITY,
			1.0,
		)
		.driving(drawn),
	);

	world.bodies.set_name(spun, "spun");
	world.dt = 1.0 / 60.0;

	let mut scripts = machine();

	stepped(&mut scripts, &mut world);

	let body = world.bodies.get(spun).expect("it is alive");

	assert!(
		(body.torque.y - 5.0).abs() < 1e-4,
		"a spin is a turn, and it is the number the program asked for: {}",
		body.torque
	);
	assert_eq!(body.force, colby_core::glam::Vec3::ZERO, "and nothing is pushing it anywhere");
}

#[test]
fn the_shipped_thruster_lifts_a_heavy_prop_slower_than_a_light_one() {
	// **the whole point of the change, and the one thing no other test here
	// can see**: what a force does depends on what the thing weighs, and that
	// only happens where a real solver runs. Two thrusters, one four times the
	// other's mass, nothing else different.
	let mut world = World::new();
	let simulation = &mut Box::new(colby_physics::Simulation::new());

	world.install_physics(simulation.table());
	world.dt = 1.0 / 60.0;
	// no gravity, so what is measured is the push rather than the difference
	// between the push and the fall - at sixteen newtons the heavy one would
	// not leave the ground at all and the test would be about that instead.
	world.gravity = colby_core::glam::Vec3::ZERO;
	world
		.sounds
		.insert("sounds/hum", colby_core::abi::SoundData {
			samples: vec![0; 48_000],
			rate: 48_000,
			channels: 1,
		});
	world
		.scripts
		.insert("scripts/thruster", ScriptData { source: THRUSTER.to_owned() });

	let mut made = |mass: f32, at: f32| {
		let drawn = world
			.entities
			.spawn_at(Transform::at(colby_core::glam::Vec3::new(at, 0.0, 0.0)));
		let it = world.bodies.spawn(
			Body::dynamic(
				Shape::cuboid(colby_core::glam::Vec3::splat(0.5)),
				Transform::at(colby_core::glam::Vec3::new(at, 0.0, 0.0)),
				mass,
			)
			.driving(drawn),
		);

		world.bodies.set_name(it, "thruster");

		it
	};

	let light = made(1.0, -8.0);
	let heavy = made(4.0, 8.0);

	let mut scripts = machine();

	stepped(&mut scripts, &mut world);
	scripts.gameplay(&mut world, &typed("thruster.on"));

	for _ in 0..30 {
		simulation.step(&mut world);
		world.bodies.end_step();
		scripts.gameplay(&mut world, &[]);
	}

	let (quick, slow) = (rising(&world, light), rising(&world, heavy));

	assert!(quick > 0.0, "the light one is climbing, at {quick}");
	assert!(
		(quick / slow - 4.0).abs() < 0.05,
		"and four times as fast as the one four times as heavy, which is what it could not do \
		 while it wrote a speed: {quick} against {slow}"
	);
}

#[test]
fn the_shipped_thruster_pushes_along_its_own_up_rather_than_the_worlds() {
	// what makes a thruster a thruster rather than a lift: turn it over and it
	// pushes the other way.
	let (mut world, it) = thrusting();
	let mut scripts = machine();

	world.dt = 1.0 / 60.0;
	let drawn = world
		.bodies
		.get(it)
		.map(|body| body.entity)
		.expect("it drives one");

	if let Some(at) = world.entities.transform_mut(drawn) {
		at.rotation = colby_core::glam::Quat::from_rotation_x(std::f32::consts::PI);
	}

	stepped(&mut scripts, &mut world);
	scripts.gameplay(&mut world, &typed("thruster.on"));

	assert!(
		pushed(&world, it) < -0.2,
		"upside down, it pushes itself into the floor: {}",
		pushed(&world, it)
	);
}

#[test]
fn the_shipped_thruster_makes_a_noise_while_it_burns_and_stops_when_it_does() {
	let (mut world, _) = thrusting();
	let mut scripts = machine();

	world.dt = 1.0 / 60.0;
	stepped(&mut scripts, &mut world);
	assert_eq!(world.audio.len(), 0, "silence until it is switched on");

	scripts.gameplay(&mut world, &typed("thruster.on"));
	assert_eq!(world.audio.len(), 1, "one thruster, one noise");

	// and a second step does not start a second one: the handle is kept in a
	// table keyed by the body, which is the whole reason a handle is an
	// ordinary value.
	scripts.gameplay(&mut world, &[]);
	assert_eq!(world.audio.len(), 1, "and still one");

	scripts.gameplay(&mut world, &typed("thruster.off"));
	assert_eq!(world.audio.len(), 0, "and none once it stops");
}

#[test]
fn the_shipped_thruster_takes_its_noise_with_it_when_it_is_removed() {
	// a noise nobody can name is a noise that plays until the process ends.
	let (mut world, it) = thrusting();
	let mut scripts = machine();

	world.dt = 1.0 / 60.0;
	stepped(&mut scripts, &mut world);
	scripts.gameplay(&mut world, &typed("thruster.on"));
	assert_eq!(world.audio.len(), 1, "it is making one");

	world.bodies.despawn(it);
	scripts.gameplay(&mut world, &[]);

	assert_eq!(world.audio.len(), 0, "and it went with the thruster");
}

#[test]
fn the_shipped_thruster_draws_itself_while_it_burns() {
	let (mut world, _) = thrusting();
	let mut scripts = machine();

	world.dt = 1.0 / 60.0;
	stepped(&mut scripts, &mut world);
	assert!(world.debug.is_empty(), "nothing is drawn while it is off");

	scripts.gameplay(&mut world, &typed("thruster.on"));

	assert!(!world.debug.lines().is_empty(), "and an arrow while it is on");
}
