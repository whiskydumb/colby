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

/// A document that names one program and holds nothing else.
fn document(program: &str) -> DocumentData {
	DocumentData {
		program: program.to_owned(),
		..DocumentData::empty()
	}
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

	scripts.update(&mut world);
	happened(&mut world, panel, "hold", EventKind::Click);
	scripts.update(&mut world);

	assert_eq!(scripts.loaded.len(), 1, "the panel is known, so it is not looked at again");
	assert!(scripts.loaded[0].handlers.is_none(), "with nothing to hand an event to");
}

#[test]
fn a_click_reaches_the_handler_that_asked_for_it() {
	let (mut world, panel) =
		showing(r#"ui.on("hold", "click", function() ui.set_text("hold", "holding") end)"#);
	let mut scripts = Vm::new().expect("the interpreter starts");

	scripts.update(&mut world);
	assert_eq!(written(&world, panel, "hold"), None, "nothing has happened yet");

	happened(&mut world, panel, "hold", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	world.ui.set_text(panel, "hold", "typed");

	happened(&mut world, panel, "hold", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	happened(&mut world, panel, "hold", EventKind::Press);
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	happened(&mut world, panel, "a", EventKind::Press);
	scripts.update(&mut world);

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
	scripts.update(&mut world);

	// what the compiler does when the file is edited: the same name, a new
	// value, and an entry whose revision has moved.
	rewritten(
		&mut world,
		r#"ui.on("hold", "click", function() ui.set_text("hold", "second") end)"#,
	);

	scripts.update(&mut world);
	happened(&mut world, panel, "hold", EventKind::Click);
	scripts.update(&mut world);

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
	scripts.update(&mut world);

	// the same name, a different document, a revision that moved.
	let mut restyled = document("ui/test");
	restyled
		.rules
		.push(colby_core::abi::ui::Rule::default());
	world.ui.insert("ui/test", restyled);

	scripts.update(&mut world);

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
	scripts.update(&mut world);

	world.ui.insert("ui/test", document("ui/other"));
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	assert!(written(&world, panel, "out").is_none(), "there was nothing to run");

	world.scripts.insert("ui/late", ScriptData {
		source: r#"ui.set_text("out", "here")"#.to_owned(),
	});
	scripts.update(&mut world);

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
	scripts.update(&mut world);
	happened(&mut world, panel, "score", EventKind::Click);
	scripts.update(&mut world);

	rewritten(&mut world, "-- nothing at all");
	scripts.update(&mut world);

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
	scripts.update(&mut world);

	rewritten(&mut world, "function (");
	scripts.update(&mut world);

	happened(&mut world, panel, "hold", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	let after = scripts.loaded[0].program_revision;
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	happened(&mut world, panel, "a", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	happened(&mut world, panel, "a", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);

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

	scripts.update(&mut world);

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

	scripts.update(&mut world);

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
		.update(&mut first);
	Vm::new()
		.expect("the interpreter starts")
		.update(&mut second);

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
	scripts.update(&mut world);
	happened(&mut world, panel, "go", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);
	happened(&mut world, panel, "a", EventKind::Click);
	scripts.update(&mut world);

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

	scripts.update(&mut world);

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
		scripts.update(&mut world);
	}

	assert_eq!(
		written(&world, panel, "out").as_deref(),
		Some("1"),
		"eight steps, one load: nothing runs a chunk again until the compiler rewrites it"
	);
	assert!(panel.is_some(), "and the panel resolved in the first place");
}
