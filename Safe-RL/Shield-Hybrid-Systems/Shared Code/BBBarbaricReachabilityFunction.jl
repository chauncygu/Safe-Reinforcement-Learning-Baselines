"""
	get_barbaric_reachability_function(samples_per_axis, mechanics)

Returns a function `R` of the signature `(square, action) -> squares::Vector{Tuple{Int}}` that computes the set of reachable squares from the initial given square and action. The function returns tuples representing squares instead of members of the Square type for performance reasons.

`samples_per_axis` specifies the number of samples that are taken along the v and p axes. The total number of samples taken will be `samples_per_axis^2`. 
"""
function get_barbaric_reachability_function(samples_per_axis, mechanics)
	return (square, action) ->
		get_reachable_area(samples_per_axis, mechanics, square, action)
end

function draw_barbaric_transition!(samples_per_axis, mechanics, square::Square, action; colors=(start=:black, _end=:gray), legend=false, plotargs...)
	v_start, p_start = [], []
	v_end, p_end = [], []
	for (v, p) in grid_points(square, samples_per_axis)
		# Start positions
		push!(v_start, v)
		push!(p_start, p)
		
		# End positions
		w, q = simulate_point(mechanics, v, p, action, unlucky=true)
		push!(v_end, w)
		push!(p_end, q)
	end
	
	
	scatter!(v_start, p_start, 
			label= legend ? "start" : nothing, 
			markersize=3, 
			markerstrokewidth=0, 
			markercolor=colors.start;
			plotargs...)
	scatter!(v_end, p_end, 
			label= legend ? "end" : nothing, 
			markersize=3, 
			markerstrokewidth=0,
			markercolor=colors._end;
			plotargs...)
	
	plot!(;plotargs...) # Pass additional arguments to Plots.jl
end


"""Get a list of grid indexes representing reachable squares. 

I could have used proper squares for this, but I want to save that extra bit of memory by not having lots of references back to the same  grid.
"""
function get_reachable_area(samples_per_axis, mechanics, square::Square, action)
	result = Set()
	for (v, p) in grid_points(square, samples_per_axis)
		w, q = simulate_point(mechanics, v, p, action, unlucky=true)
		
		if !(square.grid.v_min <= w <= square.grid.v_max) ||
			!(square.grid.p_min <= q <= square.grid.p_max)
			continue
		end
		
		square′ = box(square.grid, w, q)
		iv_ip = (square′.iv, square′.ip)
		if !(iv_ip ∈ result)
			push!(result, iv_ip)
		end
	end
	[result...]
end

"""Update the value of every square reachable from the given `square`.
"""
function set_reachable_area!(samples_per_axis, mechanics, square::Square, action, value)
	reachable_area = get_reachable_area(samples_per_axis, mechanics, square::Square, action)
	for (iv, ip) in reachable_area
		square.grid.array[iv, ip] = value
	end
end