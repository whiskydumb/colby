-- A sandbox thruster, and the first piece of gameplay in colby that is not
-- written in Rust.
--
-- What it costs is this file and `assets/props/thruster.scene` beside it. The
-- spawn menu picks the prop up on its own, because the catalogue is a walk of
-- the scene table for the `props/` prefix rather than a list anybody keeps;
-- this program is loaded on its own for the same reason, being under
-- `scripts/`. Nothing in the engine or in the game module was changed to make
-- either happen.
--
-- Type `thruster.on` at the console with a few of them spawned and frozen into
-- a shape, and the shape flies.

-- How hard one pushes, in units a second squared.
--
-- Well over the world's own 9.81 so that a thruster carrying a crate still
-- climbs, and not so far over that a bare one leaves the map in a second.
local FORCE = 16.0

-- What the prop's body is called. A prop is a `.scene` of one entity and one
-- body, and a copy keeps the names it was written with, so every thruster in
-- the world answers to this.
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
local function push(it, dt)
	local drawn = body.entity(it)
	local ux, uy, uz = entity.up(drawn)
	local px, py, pz = body.position(it)

	-- there is no way to apply a *force* to a body anywhere in this engine, so
	-- what a push is here is the speed itself, written every step. That is an
	-- impulse rather than a force and it ignores the mass, which is why a
	-- thruster lifts a heavy crate exactly as fast as a light one.
	local vx, vy, vz = body.velocity(it)
	body.set_velocity(it, vx + ux * FORCE * dt, vy + uy * FORCE * dt, vz + uz * FORCE * dt)

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
				push(it, dt)
			else
				hush(it)
			end
		end
	end

	-- and whatever was burning and is no longer in the table at all: somebody
	-- removed it, or the yard was cleaned up. A noise nobody can name is a
	-- noise that plays until the process ends.
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
