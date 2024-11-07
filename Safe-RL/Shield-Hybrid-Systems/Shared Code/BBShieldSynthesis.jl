"""Computes and returns the tuple `(hit, nohit)`.

`hit` is a 2D-array of vectors of the same layout as the `array` of the given `grid`. If a square in `grid` has index `iv, ip`, then the vector at `hit[iv, ip]` will contain tuples `(ivʹ, ipʹ)` of square indexes that are reachable by hitting the ball from `iv, ip`. 

The same goes for `nohit` just with the "nohit" action. 
"""
function get_transitions(reachability_function, grid)
	hit = Array{Vector{Any}}(undef, (grid.v_count, grid.p_count))
	nohit = Array{Vector{Any}}(undef, (grid.v_count, grid.p_count))
	
	for square in grid
		hit[square.iv, square.ip] = reachability_function(square, "hit")
		nohit[square.iv, square.ip] = reachability_function(square, "nohit")
	end
	hit, nohit
end



"""Compute the new value of a single square.

NOTE: Requires pre-baked transition matrices `reachable_hit` and `reachable_nohit`. Get these by calling `get_transitions`. 

Value can be either 0, if this square cannot reach any bad squares, 1 if the ball must be hit to avoid reaching bad squares, or 2 if reaching a bad square is inevitable.
"""
function get_new_value( reachable_hit::Matrix{Vector{Any}}, 
						reachable_nohit::Matrix{Vector{Any}}, 
						square::Square,
						grid:: Grid)
	value = get_value(square)

	if value == 2 # Bad squares stay bad. 
		return 2
	end
	
 	# check if a bad square is reachable for nohit
	nohit_bad = false
	for (iv′, ip′) in reachable_nohit[square.iv, square.ip]
		if get_value(Square(grid, iv′, ip′)) == 2
			nohit_bad = true
			break
		end
	end

	# check if hit avoids reaching bad squares
	if nohit_bad
		for (iv′, ip′) in reachable_hit[square.iv, square.ip]
			if get_value(Square(grid, iv′, ip′)) == 2
				return 2
			end
		end
		return 1
	else
		return 0
	end
end


"""Take a single step in the fixed point compuation.
"""
function shield_step( reachable_hit::Matrix{Vector{Any}}, 
	reachable_nohit::Matrix{Vector{Any}}, 
	grid::Grid)
	grid′ = Grid(grid.G, grid.v_min, grid.v_max, grid.p_min, grid.p_max)

	for square in grid
		grid′.array[square.iv, square.ip] = get_new_value(reachable_hit, reachable_nohit, square, grid)
	end
grid′
end


"""Generate shield. 

Given some initial grid, returns a tuple `(shield, terminated_early)`. 

`shield` is a new grid containing the fixed point for the given values. 

`terminted_early` is a boolean value indicating if `max_steps` were exceeded before the fixed point could be reached.
"""
function make_shield( reachable_hit::Matrix{Vector{Any}}, 
					  reachable_nohit::Matrix{Vector{Any}}, 
					  grid::Grid; 
					  max_steps=Inf,
					  animate=false,
					  colors=[:white, :powderblue, :salmon])
	animation = nothing
	if animate
		animation = Animation()
		draw(grid, colors=colors)
		frame(animation)
	end
	i = max_steps
	grid′ = nothing
	while i > 0
		grid′ = shield_step(reachable_hit, reachable_nohit, grid)
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
	(result=grid′, max_steps_reached=i==0, animation)
end


"""Generate shield. 

Given some initial grid, returns a tuple `(shield, terminated_early)`. 

`shield` is a new grid containing the fixed point for the given values. 

`terminted_early` is a boolean value indicating if `max_steps` were exceeded before the fixed point could be reached.
"""
function make_shield(reachability_function, grid::Grid;
						max_steps=Inf, 
						animate=false,
						colors=[:white, :powderblue, :salmon])
	reachable_hit, reachable_nohit = get_transitions(reachability_function, grid)
	
	return make_shield(reachable_hit, reachable_nohit, grid, 
						max_steps=max_steps, animate=animate, colors=colors)
end

function shield_action(shield::Grid, mechanics, v, p, action)
	if v < shield.v_min || v >= shield.v_max || p < shield.p_min || p >= shield.p_max
		return action
	elseif v < mechanics.v_hit || p < mechanics.p_hit
		return action
	end
	square = box(shield, v, p)
	value = get_value(square)
	if  value == 1
		return "hit"
	else
		return action
	end
end


# Check for a valid shield.
# A shield is considered invalid if...
function shield_is_valid(shield)
	# ...the red area reaches the edges of the figure.
	for square in shield
		if (square.iv == 0 || square.ip == 0 ||
			square.iv == shield.v_count || square.ip == shield.p_count)
			if get_value(square) != 0
				return false
			end
		end
	end
			
	
	#  ...the ball has a chance to start in the red area.
	lowest_starting_square = box(shield, 0, 7) # The ball starts at v=0, p ∈ [7; 10]
	if get_value(lowest_starting_square) == 2
		return false
	end

	return true
end


standard_initialization_function(Ivl, Ivu, Ipl, Ipu) = Ipl == 0 && (abs(Ivl) < 1 || abs(Ivu) < 1) ? 2 : 0