### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 78969e37-9895-4077-bdae-ff4857174c6b
begin
	using PlutoUI
	using Random
	using Plots
	include("../Shared Code/RandomWalk.jl");
end

# ╔═╡ b0b79789-e1b9-46f6-a19a-fcee9d5a5be2
md"""
# Random Walk Shielding


This is a [Pluto Notebook](https://github.com/fonsp/Pluto.jl) which can generate a shield for the **Random Walk** game. (The game is available as another notebook.)
"""

# ╔═╡ 3611edfd-a4cb-4632-9d94-2fe71e2195ae
# This simple function is used in some code cells to prevent variables from entering the global scope.
call(f) = f()

# ╔═╡ 3a43b3fe-ff43-4bf0-8bcb-6ae1781dfc41
md"""
## Color shceme

Colors by [Flat UI](https://flatuicolors.com/palette/defo)
"""

# ╔═╡ d5280fbb-cb7c-417b-816a-ded973bebb4a
begin
	colors = 
	(TURQUOISE = colorant"#1abc9c", 
	EMERALD = colorant"#2ecc71", 
	PETER_RIVER = colorant"#3498db", 
	AMETHYST = colorant"#9b59b6", 
	WET_ASPHALT = colorant"#34495e",
	
	GREEN_SEA   = colorant"#16a085", 
	NEPHRITIS   = colorant"#27ae60", 
	BELIZE_HOLE  = colorant"#2980b9", 
	WISTERIA     = colorant"#8e44ad", 
	MIDNIGHT_BLUE = colorant"#2c3e50", 
	
	SUNFLOWER = colorant"#f1c40f",
	CARROT   = colorant"#e67e22",
	ALIZARIN = colorant"#e74c3c",
	CLOUDS   = colorant"#ecf0f1",
	CONCRETE = colorant"#95a5a6",
	
	ORANGE = colorant"#f39c12",
	PUMPKIN = colorant"#d35400",
	POMEGRANATE = colorant"#c0392b",
	SILVER = colorant"#bdc3c7",
	ASBESTOS = colorant"#7f8c8d")
	
	[colors...]
end

# ╔═╡ 1b3b563a-9a3d-48fb-9c8e-34172fb80325
rwcolors = (slow=colors.PUMPKIN, fast=colors.BELIZE_HOLE, line=colors.MIDNIGHT_BLUE)

# ╔═╡ baee33de-c4b8-47a9-b231-8c7215d4b651
# I had to lighten some flatUI , though I'm sure that's against the principles of the color scheme.
shieldcolors=[colorant"#ffffff", colorant"#a1eaff", colorant"#ff9178"]

# ╔═╡ 5229f8dd-ca19-4ed0-a9d2-da1691f79089
md"""
## Set the Mechanics

Below are options to change the `mechanics` of the game. The value ϵ controls the degree of randomness, δ the average change in time (for a given action) and τ the average change in time.
"""

# ╔═╡ 2c05e965-545f-43c1-b0b0-0a81e08c6293

@bind _mechanics PlutoUI.combine() do Child

md"""
ϵ = $(Child("ϵ", NumberField(0:0.01:10, default=0.04)))

δ(:fast) = $(Child("δ_fast", NumberField(0:0.01:10, default=0.17)))
τ(:fast) = $(Child("τ_fast", NumberField(0:0.01:10, default=0.05)))

δ(:slow) = $(Child("δ_slow", NumberField(0:0.01:10, default=0.1)))
τ(:slow) = $(Child("τ_slow", NumberField(0:0.01:10, default=0.12)))
"""
	
end

# ╔═╡ 6ad63c50-77eb-4fd7-8669-085adebc0ddc
mechanics = (;_mechanics.ϵ, _mechanics.δ_fast, _mechanics.δ_slow, _mechanics.τ_fast, _mechanics.τ_slow);

# ╔═╡ 34b55cb2-e084-46af-a129-f848deacf557
md"""
The following figure illustrates the effects of the mechanics chosen above. 

The two colored areas show the possible outcomes of taking an action from the initial point.

The two lines show the worst-case run for each type of action. If the line's slope is greater than 1, it means that there is a risk that by only taking this action, the game is lost. 
"""

# ╔═╡ 715d0acd-a271-438c-a3ed-8712b024603f
md"""
The following inputs can be used to set different limits for winning/losing the game:

t\_lim = $(@bind t_lim NumberField(0:0.1:10000, default=1))
x\_lim = $(@bind x_lim NumberField(0:0.1:10000, default=1))
"""

# ╔═╡ e27afbfe-a90d-4c60-b82a-9bfc007c39fb
function fixed_cost(x, t, action)
	if action == :fast
		1
	else
		2
	end
end

# ╔═╡ 739c0741-d35d-4fc8-b93d-678371142411
function sinus_x(x, t, action)
	x = x/x_lim # Same cost no matter scaling
	a = action == :fast ? 3 : 1
	b = action == :fast ? 0 : pi
	1.5 + a + sin(b + x*pi*4)*1.5
end

# ╔═╡ 6034243b-4e9a-4720-8354-24b669bf4882
function parabola_x(x, t, action)
	x = x/x_lim # Same cost no matter scaling
	x = x - 0.5/x_lim
	a = action == :fast ? 5 : -4
	b = action == :fast ? 0.7 : 1
	a*x^2 + b
end

# ╔═╡ 06b064b9-047b-40b8-944b-ee499e1a99a1
md"""
### Costs

Several cost functions can be used to evaluate the performance of a policy. Choose a cost function below, and scroll down to see their definitions.

`cost_function`: $(@bind cost_function Select([sinus_x, parabola_x, fixed_cost]))

Regardless of the cost function, the cost of losing the game can be set with the following input:

cost\_loss = $(@bind cost_loss NumberField(0:1:10000, default=1000))
"""

# ╔═╡ a831bacb-9f95-4c94-b6ea-6e84351da678
call(() -> begin
	plot_with_size(x_lim, t_lim)
	draw_next_step!(mechanics..., 0.25, 0.25, :both, colors=rwcolors)
	wost_case_slow = take_walk(	fixed_cost, cost_loss, x_lim, t_lim, mechanics..., 
		 		(_, _) -> :slow, unlucky=true)
	wost_case_fast = take_walk(	fixed_cost, cost_loss, x_lim, t_lim, mechanics..., 
		 		(_, _) -> :fast, unlucky=true)
	draw_walk!(wost_case_slow.xs, wost_case_slow.ts, wost_case_slow.actions, colors=rwcolors)
	draw_walk!(wost_case_fast.xs, wost_case_fast.ts, wost_case_fast.actions, colors=rwcolors)
end)

# ╔═╡ f086c778-9493-4b37-baab-a95cfbf7d2e7
md"""
### Being a Cheapskate

It might be interesting to see if simply picking the cheapest option is enough to get a good score.
"""

# ╔═╡ 55b4643a-68fb-439f-b9d7-cdcf2a432589
function cheapskate(x, t)
	if cost_function(x, t, :fast) <= cost_function(x, t, :slow) 
		:fast
	else 
		:slow
	end
end

# ╔═╡ 5a1e9619-b64a-443d-8649-8cce928ae983
evaluate(cost_function, cost_loss, x_lim, t_lim, mechanics..., cheapskate)

# ╔═╡ 22ef4711-9771-4084-8157-00f817459370
md"""
## The Grid
"""

# ╔═╡ 779f0f70-ce94-4a9e-af26-3b06406aa036
md"""
### Configure grid
`G = ` $(@bind G NumberField(0.01:0.01:2, default=0.2))

`x_min:` $(@bind x_min NumberField(-100:0.1:100, default=0))
	   `x_max:`
		$(@bind x_max NumberField(-100:0.1:100, default=1.2))

`y_min:` $(@bind y_min NumberField(-100:0.1:100, default=0))
	   `y_max:`
		$(@bind y_max NumberField(-100:0.1:100, default=1.2))
"""

# ╔═╡ 53fa5c41-a5fb-4571-92ee-090ededa5cc1
call(() -> begin
	G = 0.01
	xs = [0:G:x_max;]
	fast_costs = []
	slow_costs = []
	for x in xs
		push!(fast_costs, cost_function(x, 0, :fast))
		push!(slow_costs, cost_function(x, 0, :slow))
	end
	plot(xs, fast_costs, label="fast action", c=rwcolors.fast, width=2)
	plot!(xs, slow_costs, label="slow action", c=rwcolors.slow, width=2, linestyle=:dash)
	hline!([0], label=nothing, c=:gray)
	vline!([x_lim], label=nothing, c=:gray)
	title!("Selected: $cost_function")
	xlabel!("x")
	ylabel!("cost")
end)

# ╔═╡ f8607cc8-30e5-454e-acec-6d0050a48904
begin
	struct Grid{T}
	    G::Real
	    x_min::Real
	    x_max::Real
	    y_min::Real
	    y_max::Real
	    x_count::Int
	    y_count::Int
	    array::Matrix{T}
	end
		
	function Grid(G, x_min, x_max, y_min, y_max)
	    x_count::Int = ceil((x_max - x_min)/G)
	    y_count::Int = ceil((y_max - y_min)/G)
	    array = Matrix(undef, x_count, y_count)
	    Grid(G, x_min, x_max, y_min, y_max, x_count, y_count, array)
	end
	
	Base.show(io::IO, ::MIME"text/plain", grid::Grid) = println(io, "Grid($(grid.G), $(grid.x_min), $(grid.x_max), $(grid.y_min), $(grid.y_max))")
end

# ╔═╡ 1d555d13-9b81-48e7-a74c-8e2ee388bfc2
grid = Grid(G, x_min, x_max, y_min, y_max)

# ╔═╡ 4165c794-4c2f-4d37-8a85-d1c86a32fd6c
"($(length(grid.array)) squares)"

# ╔═╡ d14ff7c8-742b-4eb2-aa04-5b1e88213f71
struct Square
    grid::Grid
    ix::Int
    iy::Int
end

# ╔═╡ c0490360-9d91-431c-8997-583c3c06b609
begin
	NEUTRAL_SQUARE = 1	# Either action allowable
	FAST_SQUARE = 2		# Must take the :fast action
	BAD_SQUARE = 3		# Risk of safety violation
	nothing
end

# ╔═╡ b06918de-37de-471f-8c6e-f5af3edcf024
init_func(Ixl, Ixu, Itl, Itu) = Itu > 1 ? BAD_SQUARE : NEUTRAL_SQUARE

# ╔═╡ 895b0abb-4ee6-4a70-b638-262583c5c8ab
function box(grid::Grid, x, y)::Square
	if x < grid.x_min || x >= grid.x_max
		error("x value out of bounds.")
	end
	if y < grid.y_min || y >= grid.y_max
		error("x value out of bounds.")
	end

	ix = floor(Int, (x - grid.x_min)/grid.G) + 1
	iy = floor(Int, (y - grid.y_min)/grid.G) + 1
	Square(grid, ix, iy)
end

# ╔═╡ 3a9bff13-e75e-4400-aefb-6ac004ca9d2e
square = box(grid, 0.5, 0.5)

# ╔═╡ d9867d36-908e-4e5e-b013-0cc0c9475982
function bounds(square::Square)
	ix, iy = square.ix-1, square.iy-1
	x_min, x_max = square.grid.x_min, square.grid.x_max
	y_min, y_max = square.grid.y_min, square.grid.y_max
	G = square.grid.G
	Ixl, Iyl = G * ix + x_min, G * iy + y_min
	Ixu, Iyu = G * (ix+1) + x_min, G * (iy+1) + y_min
	Ixl, Ixu, Iyl, Iyu
end

# ╔═╡ e1fd73cf-9651-4f94-85f9-882fd68a4ea0
function set_value!(square::Square, value)
	square.grid.array[square.ix, square.iy] = value
end

# ╔═╡ cc2f97bb-57d5-4b30-bdde-fa5e2b372c12
function get_value(square::Square)
	square.grid.array[square.ix, square.iy]
end

# ╔═╡ d85de62a-c308-4c46-9a49-5ceb37a586ba
function clear(grid::Grid)
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			grid.array[ix, iy] = 0
		end
	end
end

# ╔═╡ fe6341e8-2a52-4142-8532-52c118358c5e
function initialize!(grid::Grid, value_function=
								(Ixl, Ixu, Iyl, Iyu) -> Ixl == 0 && Iyl == 0 ? 2 : 1)
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			square = Square(grid, ix, iy)
			set_value!(square, value_function(bounds(square)...))
		end
	end
end

# ╔═╡ 2cc179a4-8848-4345-a634-bf9adca525be
initialize!(grid, init_func)

# ╔═╡ 69d9e42e-4769-46e8-8177-706d150571d8
### !! Temporary Copy-pasta !! ### 
function plot_with_size!(x_max, t_max; figure_width=600, figure_height=600)
	plot!(	xlim=[0, x_max],
			ylim=[0, t_max], 
			aspectratio=:equal, 
			xlabel="x",
			ylabel="t",
			size=(figure_width, figure_height))
	hline!([x_lim], c=:gray, label=nothing)
	vline!([t_lim], c=:gray, label=nothing)
end

# ╔═╡ d4e0a0aa-b34e-4801-9819-ea51f5b9df2a
function draw(grid::Grid; colors=[:white, :black], show_grid=false)
	colors = cgrad(colors, length(colors), categorical=true)
	x_tics = grid.x_min:grid.G:grid.x_max
	y_tics = grid.y_min:grid.G:grid.y_max
	
	hm = heatmap( 	x_tics, y_tics, 
					permutedims(grid.array, (2, 1)), 
					c=colors, 
					colorbar=nothing,
					legend=nothing,
					aspect_ratio=:equal, 
					clim=(1, length(colors)))

	if show_grid && length(grid.x_min:grid.G:grid.x_max) < 100
		vline!(grid.x_min:grid.G:grid.y_max, color="#afafaf", label=nothing)
		hline!(grid.y_min:grid.G:grid.y_max, color="#afafaf", label=nothing)
	end

	return hm
end

# ╔═╡ d92581e2-3691-4bc8-9862-aff23a75fdcc
md"""### Symbolic Transition"""

# ╔═╡ a2e85767-6f31-4c1a-a174-3bc60faf0d1b
function cover(grid, x_lower, x_upper, y_lower, y_upper)
	ix_lower = floor((x_lower - grid.x_min)/grid.G) + 1 # Julia indexes start at 1
	ix_upper = floor((x_upper - grid.x_min)/grid.G) + 1
	
	iy_lower = floor((y_lower - grid.y_min)/grid.G) + 1
	iy_upper = floor((y_upper - grid.y_min)/grid.G) + 1
	
	# Discard squares outside the grid dimensions
	ix_lower = max(ix_lower, 1)
	ix_upper = min(ix_upper, grid.x_count)
	
	iy_lower = max(iy_lower, 1)
	iy_upper = min(iy_upper, grid.y_count)
	
	[ (ix, iy)
		for ix in ix_lower:ix_upper
		for iy in iy_lower:iy_upper
	]
end

# ╔═╡ 886e8c1f-83d1-4aed-beb8-d0d73460348f
"""Get a list of grid indexes representing reachable squares. 

I could have used the square datatype for this, but I want to save that extra bit of memory by not having lots of references back to the same  grid.
"""
function get_reachable_area(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, square::Square, action)
	Ixl, Ixu, Itl, Itu = bounds(square)
	δ, τ = action == :fast ? (δ_fast, τ_fast) : (δ_slow, τ_slow)
	cover(	square.grid, 
			Ixl + δ - ϵ, 
			Ixu + δ + ϵ, 
			Itl + τ - ϵ, 
			Itu + τ + ϵ)
end

# ╔═╡ 9a0c0fbe-c450-4b42-a320-5868756a2f3d
function draw_barbaric_transition!(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, square::Square, action, resolution)
	Ixl, Ixu, Itl, Itu = bounds(square)
	δ = action == :fast ? δ_fast : δ_slow 
	τ = action == :fast ? τ_fast : τ_slow
	stride = square.grid.G/resolution
	x_start, t_start = [], []
	x_end, t_end = [], []

	for x in Ixl:stride:(Ixu)
		for t in Itl:stride:(Itu)
			push!(x_start, x)
			push!(t_start, t)
			for offset_x in (δ - ϵ):(ϵ/resolution):(δ + ϵ)
				for offset_t in (τ - ϵ):(ϵ/resolution):(τ + ϵ)
					xʹ = x + offset_x
					tʹ = t + offset_t
					push!(x_end, xʹ)
					push!(t_end, tʹ)
				end
			end
		end
	end
	scatter!(x_start, t_start, label="start", markersize=1, markerstrokewidth=0, markercolor="#888A85")
	scatter!(x_end, t_end, label="end", markersize=1, markerstrokewidth=0, markercolor="#000000")
end

# ╔═╡ a25d8cf1-1b47-4f8d-b4f7-f4e77af0ff20
	function step(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, x, t, a)
		x′, t′ =  x, t
		if a == :fast
			x′ = x + rand(δ_fast - ϵ:0.005:δ_fast + ϵ )
			t′ = t + rand(τ_fast - ϵ:0.005:τ_fast + ϵ )
		else
			x′ = x + rand(δ_slow - ϵ:0.005:δ_slow + ϵ )
			t′ = t + rand(τ_slow - ϵ:0.005:τ_slow + ϵ )
		end
		x′, t′
	end

# ╔═╡ 24d292a0-ac39-497e-b520-8fd3931369fc
function set_reachable_area!(square, t, resolution, point_function, value; upto_t=false)
	reachable_area = get_reachable_area(square, t, resolution, point_function, upto_t=upto_t)
	for (ix, iy) in reachable_area
		square.grid.array[ix, iy] = value
	end
end

# ╔═╡ 3633ff5e-19a1-4272-8c7c-5c1a3f00cc72
"""Computes and returns the tuple `(fast, slow)`.

`fast` is a 2D-array of vectors of the same layout as the `array` of the gixen `grid`. If a square in `grid` has index `ix, iy`, then the vector at `fst[ix, iy]` will contain tuples `(ixʹ, iyʹ)` of square indexes that are reachable by performing a fast move from the given square.

The same goes for `slow` just with the :slow action. 
"""
function get_transitions(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, grid)
	fast = Matrix{Vector{Any}}(undef, grid.x_count, grid.y_count)
	slow = Matrix{Vector{Any}}(undef, grid.x_count, grid.y_count)
	
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			square = Square(grid, ix, iy)
			fast[ix, iy] = get_reachable_area(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, square, :fast)
			slow[ix, iy] = get_reachable_area(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, square, :slow)
		end
	end
	fast, slow
end

# ╔═╡ 4fa89f9a-7aa7-441c-99a5-4be7b1055bbe
fast, slow = get_transitions(mechanics..., grid);

# ╔═╡ 7abee61b-36bc-488b-893e-d42b5ca8665d
mechanics

# ╔═╡ 795c5353-fdeb-41c6-8502-2aa70689dcc4
#Compute the new value of a single square.
#
#NOTE: Requires pre-baked transition matrices. Get these by calling `get_transitions`.
function get_new_value(fast::Matrix{Vector{Any}}, slow::Matrix{Vector{Any}}, square)
	value = get_value(square)

	if value == BAD_SQUARE # Bad squares stay bad. 
		return BAD_SQUARE
	end

	Ixl, Ixu, Itl, Itu = bounds(square)

	if Ixl >= 1 && Itu <= 1 
		return NEUTRAL_SQUARE
	end

	# Check if a bad square is reachable while going slow.
	slow_bad = false
	for (ix, iy) in slow[square.ix, square.iy]
		if get_value(Square(square.grid, ix, iy)) == BAD_SQUARE
			slow_bad = true
			break
		end
	end

	if !slow_bad
		# Assuming fast is better than slow, this means both actions are allowable
		return NEUTRAL_SQUARE 
	end

	# Check if bad squares can be avoided by going fast.
	fast_bad = false
	for (ix, iy) in fast[square.ix, square.iy]
		if get_value(Square(square.grid, ix, iy)) == BAD_SQUARE
			fast_bad = true
			break
		end
	end

	if !fast_bad
		return FAST_SQUARE # Gotta go fast.
	else
		return BAD_SQUARE
	end
end

# ╔═╡ 340dcc48-3787-4894-85aa-0d13873d19db
(;NEUTRAL_SQUARE, FAST_SQUARE, BAD_SQUARE)

# ╔═╡ c2add912-1322-4f34-b9d5-e2284f631b3c
get_new_value(fast, slow, square)

# ╔═╡ c96bd5bc-6f4a-43db-a3d0-892b0f960cc4
begin
	# Check if a bad square is reachable while going slow.
	slow_bad = false
	for (ix, iy) in slow[square.ix, square.iy]
		if get_value(Square(square.grid, ix, iy)) == BAD_SQUARE
			slow_bad = true
			break
		end
	end
	
	# Check if bad squares can be avoided by going fast.
	fast_bad = false
	for (ix, iy) in fast[square.ix, square.iy]
		if get_value(Square(square.grid, ix, iy)) == BAD_SQUARE
			fast_bad = true
			break
		end
	end
	(;slow_bad, fast_bad)
end

# ╔═╡ f9b7b12f-5193-48ec-b61c-ba22f4a1fb4c
# Take a single step in the fixed point compuation.
function shield_step(fast::Matrix{Vector{Any}}, slow::Matrix{Vector{Any}}, grid)
	grid′ = Grid(grid.G, grid.x_min, grid.x_max, grid.y_min, grid.y_max)
	
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			grid′.array[ix, iy] = get_new_value(fast, slow, Square(grid, ix, iy))
		end
	end
	grid′
end

# ╔═╡ 6c7b61f9-98d5-4f7b-b88d-8f74ca1bbcb3
function make_shield(   ϵ, δ_fast, δ_slow, τ_fast, τ_slow, grid;
					    max_steps=1000, 
						animate=false)
	transitions = get_transitions(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, grid)
	
	return make_shield(transitions..., grid, max_steps=max_steps, animate=animate)
end

# ╔═╡ 18b843fd-2ab8-4380-a700-240115dd23da
md"""
#### Shield config
`max_steps =` $(@bind max_steps NumberField(0:1:1000, default=100))

`animate ` $(@bind animate html"<input type=checkbox>")

`fps =` $(@bind fps NumberField(0:1:1000, default=3))
"""

# ╔═╡ be817e03-90a1-45ec-b5dd-04ab4ca2aaa9
begin
	cheapskate_animation = 
		@animate for i in 1:50
			call(() -> begin 
				plot_with_size(x_lim, t_lim)
				xs, ts, actions, total_cost, winner =  take_walk(	
					fixed_cost, cost_loss, x_lim, t_lim, mechanics..., 
					cheapskate, unlucky=false)
				draw_walk!(xs, ts, actions, colors=rwcolors)
				title!("Picking the cheapest cost for $(cost_function):")
			end)
		end
	gif(cheapskate_animation, "cheapskate.gif", fps=fps)
end

# ╔═╡ fc2dafd2-aea5-49c9-92d3-f7b478be3be0
md"""
show grid: $(@bind show_grid CheckBox(default=true))

show step: $(@bind show_step CheckBox(default=true))
"""

# ╔═╡ be4a5a08-79b8-4ac9-8396-db5d62eb3f97
if show_step 
md"""
#### Example values for x and t

`x = ` $(@bind x NumberField(0.01:0.01:4))
`t = ` $(@bind t NumberField(0.01:0.01:4))
"""
end

# ╔═╡ f1e3d59d-4a15-43c1-8717-b736c90e127c
# Remember to set a plot size first, otherwise it don't work
function show_legend!(;colors)
	scatter!([-100],[-100], color=shieldcolors[1], markersize=5, markershape=:circle, legend=:topleft, labels="{slow, fast}")
	scatter!([-100],[-100], color=shieldcolors[2], markersize=5, markershape=:circle, legend=:topleft, labels="{fast}")
	scatter!([-100],[-100], color=shieldcolors[3], markersize=5, markershape=:circle, legend=:topleft, labels="∅")
end

# ╔═╡ f09c6e8a-b1b9-4832-a632-bd2c4e60d8aa
call(() -> begin
	grid = Grid(G, x_min, x_max, y_min, y_max)
	initialize!(grid, init_func)
	draw(grid, colors=shieldcolors, show_grid=true)
	plot_with_size!(x_max, y_max)
	show_legend!(colors=shieldcolors)
end)

# ╔═╡ e7c8c0fe-8008-4ae0-abc8-c2cb3eb711d9
#Generate shield. 
#
#Gixen some initial grid, returns a tuple `(shield, terminated_early)`. 
#
#`shield` is a new grid containing the fixed point for the gixen values. 
#
#`terminted_early` is a boolean value indicating if `max_steps` were exceeded before #the fixed point could be reached.
#
function make_shield(	fast::Matrix{Vector{Any}}, slow::Matrix{Vector{Any}}, grid; 
					 	max_steps=1000,
					  	animate=false,
						colors=[:white, :blue, :red])
	animation = nothing
	if animate
		animation = Animation()
		draw(grid, colors=colors, show_grid=true)
		plot_with_size!(x_max, y_max)
		show_legend!(colors=colors)
		frame(animation)
	end
	i = max_steps
	grid′ = nothing
	while i > 0
		grid′ = shield_step(fast, slow, grid)
		if grid′.array == grid.array
			break
		end
		grid = grid′
		i -= 1
		if animate
			draw(grid, colors=colors, show_grid=true)
			plot_with_size!(x_max, y_max)
			show_legend!(colors=colors)
			frame(animation)
		end
	end
	(grid=grid′, terminated_early=i==0, animation)
end

# ╔═╡ b00bbbb7-6587-4664-ae82-82c081f66f37
begin
	initialize!(grid, init_func)
	if max_steps == 0
		shield, finished_early, animation = grid, true, nothing
	else
		
		#### Make the shield! ###
		shield, finished_early, animation = 
			make_shield(fast, slow, grid, max_steps=max_steps, colors=shieldcolors, animate=animate)
		
	end
end

# ╔═╡ 8a027b31-a064-46f3-8a76-ce3184eade26
if finished_early
Markdown.parse("""
!!! warning

	Shield not done!
	Computation terminated before a fixed point was reached. 

	Increase `max_steps` further if you wish to see the finished shield.
""")
end

# ╔═╡ cb460b6d-aa08-4472-bab9-737c89e2224f
begin
	shieldplot = draw(shield, colors=shieldcolors, show_grid=show_grid)
	plot_with_size!(x_max, y_max)
	show_legend!(colors=shieldcolors)
	if show_step
		draw_next_step!(mechanics..., x, t, :both, colors=rwcolors)
	end
	shieldplot
end

# ╔═╡ 896993db-f8d4-492b-bff1-463658587a83
animation != nothing ? gif(animation, "shield.gif", fps=fps) : nothing

# ╔═╡ 397ca36e-bd4a-45da-9f26-573c10a938fa
function shield_action(shield, x, t, action)
	if x < shield.x_min || x >= shield.x_max || t < shield.y_min || t >= shield.y_max
		return action
	end
    square_value = get_value(box(shield, x, t))
	if square_value == FAST_SQUARE || square_value == BAD_SQUARE
		return :fast
	else
		return action
	end
end

# ╔═╡ 97962767-65eb-4b22-80bb-e352ec60e3e8
shielded_layabout = (x, t) -> shield_action(shield, x, t, :slow);

# ╔═╡ 6387760b-9c16-4ab0-8229-07f084d2b050
xs, ts, actions, total_cost, winner = 
	take_walk(fixed_cost, cost_loss, x_lim, t_lim, mechanics..., shielded_layabout, unlucky=true)

# ╔═╡ 7c911e4c-e132-473e-a579-c47c0b348e6c
begin
	draw(shield, colors=shieldcolors)
	draw_walk!(xs, ts, actions, colors=rwcolors)
	plot_with_size!(x_max, y_max)
	plot!(title="Worst-case walk under shield")
end

# ╔═╡ 4175fe77-c75f-4c2e-a23f-3c37ac8c2f1d
evaluate(cost_function, cost_loss, x_lim, t_lim, mechanics..., shielded_layabout, iterations=1000)

# ╔═╡ 45aabb2b-6a8f-462c-b082-7d7675676d64
begin
	shielded_walks_animation = 
		@animate for i in 1:50
			call(() -> begin 
				draw(shield, colors=shieldcolors)
				plot_with_size!(x_max, y_max)
				xs, ts, actions, total_cost, winner =  take_walk(	
					fixed_cost, cost_loss, x_lim, t_lim, mechanics..., 
					shielded_layabout, unlucky=false)
				draw_walk!(xs, ts, actions, colors=rwcolors)
			end)
		end
	gif(shielded_walks_animation, "shielded_walks.gif", fps=fps)
end

# ╔═╡ b0f5055d-36fe-4593-bdfa-f6968dbb8242
md"""
**Make C header:** $(@bind make_c_header CheckBox(default=false))

Generates a header snippet for the C shielding library, to import the shield into UPPAAL.
"""

# ╔═╡ aa8a34cb-fd4e-4afc-87a3-7d92d12a25a1
function stringdump(grid)
	result = Vector{Char}(repeat('?', grid.x_count*grid.y_count))
	for it in 1:grid.y_count
		for ix in 1:grid.x_count
			square_value = get_value(Square(grid, ix, it))
			if square_value == BAD_SQUARE
				result[ix + (it-1)*grid.x_count] = 'r'
			elseif square_value == FAST_SQUARE
				result[ix + (it-1)*grid.x_count] = 'b'
			else
				result[ix + (it-1)*grid.x_count] = 'w'
			end
		end
	end
	return String(result)
end

# ╔═╡ c020277d-2dea-4879-90f7-85f7f85d1e3b
function get_c_library_header(grid)
	"""
	/* This code was automatically generated by function get_c_library_header*/
	char grid[] = "$(stringdump(grid))";
    const double G = $(grid.G);
    const double x_min = $(grid.x_min);
    const double x_max = $(grid.x_max);
    const double y_min = $(grid.y_min);
	const double y_max = $(grid.y_max);
	// Mechanics: $mechanics"""
end

# ╔═╡ fd0289fb-b463-4bb5-a775-597f011d8a36
if make_c_header
	Markdown.parse(string(get_c_library_header(shield)))
else
	md"(C header will not be generated)"
end

# ╔═╡ 27e3cdb0-59b8-4728-bbd9-da606763d18e
shield_action(shield, 0.1, 0.9, :slow)

# ╔═╡ 6c4c07b7-89f0-4cc6-9d60-ab51ef9fe566
shield_action(shield, 1.0, 0.0, :slow)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Plots = "~1.27.3"
PlutoUI = "~0.7.37"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "cb6b1762d5ca8a3e1791fa28a09068ac94f9c74d"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "df5f5b0450c489fe6ed59a6c0a9804159c22684d"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.1"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "83578392343a7885147726712523c39edc714956"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.1+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
git-tree-sha1 = "737a5957f387b17e74d4ad2f440eb330b39a62c5"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5f6e1309595e95db24342e56cd4dabd2159e0b79"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─b0b79789-e1b9-46f6-a19a-fcee9d5a5be2
# ╠═78969e37-9895-4077-bdae-ff4857174c6b
# ╠═3611edfd-a4cb-4632-9d94-2fe71e2195ae
# ╟─3a43b3fe-ff43-4bf0-8bcb-6ae1781dfc41
# ╟─d5280fbb-cb7c-417b-816a-ded973bebb4a
# ╟─1b3b563a-9a3d-48fb-9c8e-34172fb80325
# ╟─baee33de-c4b8-47a9-b231-8c7215d4b651
# ╟─5229f8dd-ca19-4ed0-a9d2-da1691f79089
# ╟─2c05e965-545f-43c1-b0b0-0a81e08c6293
# ╟─6ad63c50-77eb-4fd7-8669-085adebc0ddc
# ╟─34b55cb2-e084-46af-a129-f848deacf557
# ╟─a831bacb-9f95-4c94-b6ea-6e84351da678
# ╟─715d0acd-a271-438c-a3ed-8712b024603f
# ╟─06b064b9-047b-40b8-944b-ee499e1a99a1
# ╟─53fa5c41-a5fb-4571-92ee-090ededa5cc1
# ╠═e27afbfe-a90d-4c60-b82a-9bfc007c39fb
# ╠═739c0741-d35d-4fc8-b93d-678371142411
# ╠═6034243b-4e9a-4720-8354-24b669bf4882
# ╟─f086c778-9493-4b37-baab-a95cfbf7d2e7
# ╟─55b4643a-68fb-439f-b9d7-cdcf2a432589
# ╟─be817e03-90a1-45ec-b5dd-04ab4ca2aaa9
# ╠═5a1e9619-b64a-443d-8649-8cce928ae983
# ╟─22ef4711-9771-4084-8157-00f817459370
# ╟─779f0f70-ce94-4a9e-af26-3b06406aa036
# ╠═1d555d13-9b81-48e7-a74c-8e2ee388bfc2
# ╟─4165c794-4c2f-4d37-8a85-d1c86a32fd6c
# ╠═f8607cc8-30e5-454e-acec-6d0050a48904
# ╠═d14ff7c8-742b-4eb2-aa04-5b1e88213f71
# ╠═3a9bff13-e75e-4400-aefb-6ac004ca9d2e
# ╠═c0490360-9d91-431c-8997-583c3c06b609
# ╠═b06918de-37de-471f-8c6e-f5af3edcf024
# ╠═2cc179a4-8848-4345-a634-bf9adca525be
# ╠═f09c6e8a-b1b9-4832-a632-bd2c4e60d8aa
# ╟─895b0abb-4ee6-4a70-b638-262583c5c8ab
# ╟─d9867d36-908e-4e5e-b013-0cc0c9475982
# ╟─e1fd73cf-9651-4f94-85f9-882fd68a4ea0
# ╟─cc2f97bb-57d5-4b30-bdde-fa5e2b372c12
# ╟─d85de62a-c308-4c46-9a49-5ceb37a586ba
# ╟─fe6341e8-2a52-4142-8532-52c118358c5e
# ╟─69d9e42e-4769-46e8-8177-706d150571d8
# ╠═d4e0a0aa-b34e-4801-9819-ea51f5b9df2a
# ╟─d92581e2-3691-4bc8-9862-aff23a75fdcc
# ╠═886e8c1f-83d1-4aed-beb8-d0d73460348f
# ╠═a2e85767-6f31-4c1a-a174-3bc60faf0d1b
# ╟─9a0c0fbe-c450-4b42-a320-5868756a2f3d
# ╟─a25d8cf1-1b47-4f8d-b4f7-f4e77af0ff20
# ╟─24d292a0-ac39-497e-b520-8fd3931369fc
# ╟─3633ff5e-19a1-4272-8c7c-5c1a3f00cc72
# ╠═4fa89f9a-7aa7-441c-99a5-4be7b1055bbe
# ╠═7abee61b-36bc-488b-893e-d42b5ca8665d
# ╠═795c5353-fdeb-41c6-8502-2aa70689dcc4
# ╟─340dcc48-3787-4894-85aa-0d13873d19db
# ╠═c2add912-1322-4f34-b9d5-e2284f631b3c
# ╠═c96bd5bc-6f4a-43db-a3d0-892b0f960cc4
# ╠═f9b7b12f-5193-48ec-b61c-ba22f4a1fb4c
# ╠═e7c8c0fe-8008-4ae0-abc8-c2cb3eb711d9
# ╠═6c7b61f9-98d5-4f7b-b88d-8f74ca1bbcb3
# ╟─18b843fd-2ab8-4380-a700-240115dd23da
# ╠═b00bbbb7-6587-4664-ae82-82c081f66f37
# ╟─8a027b31-a064-46f3-8a76-ce3184eade26
# ╟─fc2dafd2-aea5-49c9-92d3-f7b478be3be0
# ╟─be4a5a08-79b8-4ac9-8396-db5d62eb3f97
# ╟─f1e3d59d-4a15-43c1-8717-b736c90e127c
# ╠═cb460b6d-aa08-4472-bab9-737c89e2224f
# ╟─896993db-f8d4-492b-bff1-463658587a83
# ╠═397ca36e-bd4a-45da-9f26-573c10a938fa
# ╠═97962767-65eb-4b22-80bb-e352ec60e3e8
# ╠═6387760b-9c16-4ab0-8229-07f084d2b050
# ╟─7c911e4c-e132-473e-a579-c47c0b348e6c
# ╠═4175fe77-c75f-4c2e-a23f-3c37ac8c2f1d
# ╟─45aabb2b-6a8f-462c-b082-7d7675676d64
# ╟─b0f5055d-36fe-4593-bdfa-f6968dbb8242
# ╟─aa8a34cb-fd4e-4afc-87a3-7d92d12a25a1
# ╟─c020277d-2dea-4879-90f7-85f7f85d1e3b
# ╟─fd0289fb-b463-4bb5-a775-597f011d8a36
# ╠═27e3cdb0-59b8-4728-bbd9-da606763d18e
# ╠═6c4c07b7-89f0-4cc6-9d60-ab51ef9fe566
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
