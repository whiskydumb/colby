-- A thruster: the program the tests of the world-program API run.
--
-- A complete program rather than a probe, kept as a file rather than a string
-- inside a test so that it reads the way a program somebody writes reads and
-- so that a mutation driver can point at it. It began as a game's own prop:
-- every body called `thruster` is pushed along its own up while the thrusters
-- are switched on, makes a noise while it burns and draws an arrow, and the
-- three commands below are what switch them. Put it under a project's
-- `assets/scripts/` and the host runs it on its own, being under `scripts/`.
--
-- Type `thruster.on` at the console with a few of them in the world, and they
-- fly.

-- How hard one pushes, in newtons.
--
-- **Newtons rather than units a second squared**, which is what this number
-- was until a body could be handed a force: it is now divided by whatever the
-- thing weighs, so one thruster lifts its own one-kilogram body briskly and
-- two of them are needed to lift it and a crate. That is the point of the
-- change and it is a change in behavior, not only in units.
--
-- Sixteen is well over the 9.81 a kilogram needs to hover, and not so far over
-- that a bare thruster leaves the map in a second.
local FORCE = 16.0

-- What a thruster's body is called. A scene of one entity and one body keeps
-- the names it was written with when it is put into a world, so every copy of
-- one answers to this.
local NAME = "thruster"

-- How long the arrow drawn out of one is, in units.
local ARROW = 1.6

-- Whether they are burning.
local burning = false

-- The noise each one is making, keyed by the body itself.
--
-- **A handle is an ordinary value here**: it goes in a table as a key, comes
-- back out equal to itself, and two of them naming one body are one key. That
-- is the whole reason a handle crosses as a tagged number rather than as an
-- object.
local noise = {}

-- The ones that were burning last step, in the order the table holds them.
--
-- An array beside the lookup above, and not for speed: walking a Lua table with
-- `pairs` visits it in whatever order the interpreter's string seed decided, so
-- a program that did anything in that order would make two runs of a screenshot
-- two pictures. Everything the engine hands over is in slot order, and this
-- keeps it that way.
local burned = {}

colby.publish("thruster.on", "switch every thruster on", function()
	burning = true
end)

colby.publish("thruster.off", "switch every thruster off", function()
	burning = false
end)

colby.publish("thruster.toggle", "switch every thruster over", function()
	burning = not burning
end)

-- Pushes one thruster along its own up, and says so.
local function push(it)
	local drawn = body.entity(it)
	local ux, uy, uz = entity.up(drawn)
	local px, py, pz = body.position(it)

	-- a force rather than a speed, so what this does depends on what the
	-- thing weighs. The engine divides by the mass and spends it over the step,
	-- which is why `dt` is no longer in this line: a force is a rate already.
	-- @ref `Bodies::apply_force`.
	body.push(it, ux * FORCE, uy * FORCE, uz * FORCE)

	draw.arrow(px, py, pz, px - ux * ARROW, py - uy * ARROW, pz - uz * ARROW, 1.0, 0.5, 0.1)

	if noise[it] then
		sound.move(noise[it], px, py, pz)
	else
		noise[it] = sound.play("sounds/hum", px, py, pz, true)
	end
end

-- Stops the noise one was making, if it was making one.
local function hush(it)
	if noise[it] then
		sound.stop(noise[it])
		noise[it] = nil
	end
end

function tick(dt)
	local all = body.all()
	local now = {}

	for i = 1, #all do
		local it = all[i]

		if body.name(it) == NAME then
			now[#now + 1] = it

			if burning then
				push(it)
			else
				hush(it)
			end
		end
	end

	-- and whatever was burning and is no longer in the table at all: somebody
	-- removed it. A noise nobody can name is a noise that plays until the
	-- process ends.
	for i = 1, #burned do
		local was = burned[i]
		local still = false

		for j = 1, #now do
			if now[j] == was then
				still = true
				break
			end
		end

		if not still then
			hush(was)
		end
	end

	burned = now
end
