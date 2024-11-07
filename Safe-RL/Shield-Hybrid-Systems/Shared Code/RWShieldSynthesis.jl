NEUTRAL_SQUAREʹ = 1	# Either action allowable
FAST_SQUAREʹ = 2		# Must take the :fast action
BAD_SQUAREʹ = 3		# Risk of safety violation


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


function set_reachable_area!(square, t, resolution, point_function, value; upto_t=false)
	reachable_area = get_reachable_area(square, t, resolution, point_function, upto_t=upto_t)
	for (ix, iy) in reachable_area
		square.grid.array[ix, iy] = value
	end
end


"""Computes and returns the tuple `(fast, slow)`.

`fast` is a 2D-array of vectors of the same layout as the `array` of the gixen `grid`. If a square in `grid` has index `ix, iy`, then the vector at `fst[ix, iy]` will contain tuples `(ixʹ, iyʹ)` of square indexes that are reachable by performing a fast move from the given square.

The same goes for `slow` just with the :slow action. 
"""
function get_transitions(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, grid)
	fast = Matrix{Vector{Any}}(undef, grid.x_count, grid.y_count)
	slow = Matrix{Vector{Any}}(undef, grid.x_count, grid.y_count)
	
	@progress for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			square = Square(grid, ix, iy)
			fast[ix, iy] = get_reachable_area(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, square, :fast)
			slow[ix, iy] = get_reachable_area(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, square, :slow)
		end
	end
	fast, slow
end


"""Compute the new value of a single square.

NOTE: Requires pre-baked transition matrices. Get these by calling `get_transitions`. 
"""
function get_new_value(fast::Matrix{Vector{Any}}, slow::Matrix{Vector{Any}}, square)
	value = get_value(square)

	if value == BAD_SQUAREʹ # Bad squares stay bad. 
		return BAD_SQUAREʹ
	end

	Ixl, Ixu, Itl, Itu = bounds(square)

	if Ixl >= 1 && Itu <= 1 
		return NEUTRAL_SQUAREʹ
	end

	# Check if a bad square is reachable while going slow.
	slow_bad = false
	for (ix, iy) in slow[square.ix, square.iy]
		if get_value(Square(square.grid, ix, iy)) == BAD_SQUAREʹ
			slow_bad = true
			break
		end
	end

	if !slow_bad
		# Assuming fast is better than slow, this means both actions are allowable
		return NEUTRAL_SQUAREʹ 
	end

	# Check if bad squares can be avoided by going fast.
	fast_bad = false
	for (ix, iy) in fast[square.ix, square.iy]
		if get_value(Square(square.grid, ix, iy)) == BAD_SQUAREʹ
			fast_bad = true
			break
		end
	end

	if !fast_bad
		return FAST_SQUAREʹ # Gotta go fast.
	else
		return BAD_SQUAREʹ
	end
end


"""Take a single step in the fixed point compuation.
"""
function shield_step(fast::Matrix{Vector{Any}}, slow::Matrix{Vector{Any}}, grid)
	grid′ = Grid(grid.G, grid.x_min, grid.x_max, grid.y_min, grid.y_max)
	
	for ix in 1:grid.x_count
		for iy in 1:grid.y_count
			grid′.array[ix, iy] = get_new_value(fast, slow, Square(grid, ix, iy))
		end
	end
	grid′
end


"""Generate shield. 

Gixen some initial grid, returns a tuple `(shield, terminated_early)`. 

`shield` is a new grid containing the fixed point for the gixen values. 

`terminted_early` is a boolean value indicating if `max_steps` were exceeded before the fixed point could be reached.
"""
function make_shield(	fast::Matrix{Vector{Any}}, slow::Matrix{Vector{Any}}, grid; 
					 	max_steps=1000,
					  	animate=false,
						colors=[:white, :blue, :red])
	animation = nothing
	if animate
		animation = Animation()
		draw(grid, colors=colors)
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
			draw(grid, colors=colors)
			frame(animation)
		end
	end
	(grid=grid′, terminated_early=i==0, animation)
end


function make_shield(   ϵ, δ_fast, δ_slow, τ_fast, τ_slow, grid;
					    max_steps=1000, 
						animate=false,
						colors=[:white, :blue, :red])
	transitions = get_transitions(ϵ, δ_fast, δ_slow, τ_fast, τ_slow, grid)
	
	return make_shield(transitions..., grid, max_steps=max_steps, animate=animate, colors=colors)
end


function shield_action(shield, x, t, action)
	if x < shield.x_min || x >= shield.x_max || t < shield.y_min || t >= shield.y_max
		return action
	end
    square_value = get_value(box(shield, x, t))
	if square_value == FAST_SQUAREʹ || square_value == BAD_SQUAREʹ
		return :fast
	else
		return action
	end
end


function get_shielding_function(shield::Grid, strategy)
	(x, t) -> shield_action(shield, x, t, strategy(x, t))
end