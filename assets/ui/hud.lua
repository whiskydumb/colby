-- the demo's interface logic, and the whole of what Lua does in colby today.
--
-- Edit this file while the engine is running and the buttons change behavior in
-- the window: the compiler turns it into `ui/hud.clua`, which is an asset of its
-- own like a mesh, `hud.html` names it with `<script src="ui/hud">` the way an
-- image names a texture, and this program is replaced with the gameplay module
-- never being touched. Nothing here survives that -- `swept` below starts at
-- zero again -- while whatever the program wrote into the panel does, because
-- that is the host's. Editing `theme.css` beside it does not restart it at all,
-- which is what the program being its own asset bought.
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

-- and the button that remembers something itself. How many times the yard has
-- been cleared is this program's business, the words and the color are its
-- business too, and the only part the game is told about is that the props
-- should go. Editing this file replaces the program and `swept` starts at zero
-- again; a reload of the *gameplay* module does not, and neither does a change
-- to the document or the stylesheet beside it.
local swept = 0

ui.on("cleanup", "click", function()
	swept = swept + 1

	colby.command("game.cleanup")
	ui.set_text("cleanup", "cleanup " .. swept)
	ui.set_classes("cleanup", "button on")
end)
