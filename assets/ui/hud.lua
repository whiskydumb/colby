-- the demo's interface logic, and the whole of what Lua does in colby today.
--
-- Edit this file while the engine is running and the buttons change behavior in
-- the window: the compiler folds it into `ui/hud.cdoc` the same way it folds
-- `theme.css` in, the runner reloads the document, and this program is replaced
-- with the gameplay module never being touched. Nothing here survives that --
-- `holding` below starts false again -- while whatever the program wrote into
-- the panel does, because that is the host's.
--
-- What is reachable from here is `ui` and one way out, `colby.command`. There
-- is no entity, no body and no camera in this environment, and no clock either:
-- interface logic is what a script does, and gameplay stays in Rust.

-- the button that asks the game for something. `game.reset` is a console
-- command `colby_game` registers in its own init, so this reaches gameplay
-- through a name the game chose to publish rather than through its state.
ui.on("reset", "click", function()
	colby.command("game.reset")
end)

-- and the button that remembers something itself. Whether it is pressed is this
-- program's business, the words and the color are its business too, and the
-- only part the game is told about is that the ring should stop turning.
local holding = false

ui.on("hold", "click", function()
	holding = not holding

	colby.command("game.hold")
	ui.set_text("hold", holding and "holding" or "hold")
	ui.set_classes("hold", holding and "button on" or "button")
end)
