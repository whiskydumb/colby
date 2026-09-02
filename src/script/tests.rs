//! What a document's script may and may not do, checked without a window.
//!
//! Everything here builds a `World` by hand, registers a program and a document
//! naming it, and drives [`Vm::update`] the way the step does. No GPU, no files
//! and no interpreter state left over between tests: each one gets its own
//! [`Vm`], which is also the only way to be sure two documents are isolated
//! because they are given separate environments rather than because nothing has
//! collided yet.

use colby_core::abi::{
	ScriptData, Value, World,
	ui::{DocumentData, Event, EventKind, PanelId},
};

use super::Vm;

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

/// One whole step's worth of the interpreter, which is two calls.
///
/// The step drives the two halves at two different moments - the interface
/// before the physics and the world's own after the game's `update` - and
/// almost every test here is about what one step does, so this is the fixture
/// rather than either half on its own.
fn stepped(scripts: &mut Vm, world: &mut World) {
	scripts.interface(world);
	scripts.gameplay(world);
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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

	Vm::new()
		.expect("the interpreter starts")
		.interface(&mut first);
	Vm::new()
		.expect("the interpreter starts")
		.interface(&mut second);

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

	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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

	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("nil"), "there is no `ui` in the environment");
}

#[test]
fn editing_a_world_program_runs_it_again() {
	let mut world = running(r#"colby.command("script.said first")"#);
	listen(&mut world);
	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");
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

	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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

	let mut scripts = Vm::new().expect("the interpreter starts");
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
	let mut scripts = Vm::new().expect("the interpreter starts");

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

	let mut scripts = Vm::new().expect("the interpreter starts");
	stepped(&mut scripts, &mut world);

	assert_eq!(said(&world).as_deref(), Some("0.25"), "which is what the step is on");
}

#[test]
fn a_program_that_wants_no_tick_is_not_called_and_costs_nothing() {
	let mut world = running("-- a program that runs once and is done");
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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
	let mut scripts = Vm::new().expect("the interpreter starts");

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

	let mut scripts = Vm::new().expect("the interpreter starts");
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

	let mut scripts = Vm::new().expect("the interpreter starts");
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
