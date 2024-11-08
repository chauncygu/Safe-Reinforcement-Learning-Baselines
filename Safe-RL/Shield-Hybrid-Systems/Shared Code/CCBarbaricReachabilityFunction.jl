#=
	using Plots
	using Serialization
	using Random
	using StatsBase
	using Unzip
=#  

function get_cc_grid(G, mechanics::CCMechanics)
	Grid(G, [
			mechanics.v_ego_min,
			mechanics.v_front_min - 1, # Front can exceed speed limits by 1
			-1 # I think I clamp distance to -1 somewhere
		], [
			mechanics.v_ego_max + G, # + G to include the upper bound
			mechanics.v_front_max + 1 + G, # Front can exceed speed limits by 1
			mechanics.ego_sensor_range + 1 + G # distance is clamped to 201
		])
end


# Iterable showing possible front actions. 
# The behaviour of the uncontrollable player is seen as additional dimensions when approximating reachability.
# In the case where the action space is a set of possible integers, samples are taken at regular intervals just as with other dimensions. When actions are discrete, all actions are sampled.
struct possible_front_actions
	samples_per_axis::Int
	mechanics::CCMechanics
	state::Vector{Number}
end

Base.length(iter::possible_front_actions) = begin
	if iter.state[3] > iter.mechanics.ego_sensor_range
		return iter.samples_per_axis
	else
		return length(instances(CCAction))
	end
end

Base.iterate(iter::possible_front_actions) = begin
	if iter.state[3] <= iter.mechanics.ego_sensor_range
		
		# Initiate the iterator whose state is type CCAction
		return backwards, backwards
	else
		# An action equal to or greater than ego's speed means staying out of sensor range.
		lower = iter.mechanics.v_front_min - 1 # Due to a bug, front can exceed speed limits by 1
		upper = iter.state[1]
		
		spacing = (upper - lower)/(iter.samples_per_axis - 1)
		
		# Initiate the iterator whose state is type NamedTuple
		return round(Int, lower), (spacing, lower, i=0)
	end
end

Base.iterate(iter::possible_front_actions, iterstate::NamedTuple) = begin
	spacing, lower, i = iterstate
	if i < iter.samples_per_axis - 1
		i += 1
		return round(Int, i*spacing + lower), (;spacing, lower,  i)
	else 
		return nothing
	end
end

Base.iterate(iter::possible_front_actions, iterstate::CCAction) = begin
	if iterstate != forwards
		action = iterstate
		action += 2
		return action, action
	else
		return nothing
	end
end
    

# Returns a list of states that are possible outcomes from the initial square
# sampled as regularly spaced grid-points defined from samples_per_axis
function possible_outcomes(samples_per_axis, mechanics::CCMechanics, 
			square::Square, action::CCAction)
	
	result = []
	for s in grid_points(square, samples_per_axis)
		for front_action in possible_front_actions(samples_per_axis, mechanics, s)
			s′ = simulate_point(mechanics, front_action, s, action)
			s′ = (s′[1], s′[2], max(-1., s′[3]))
			push!(result, s′)
		end
	end
	result
end

"""Get a list of grid indexes representing reachable squares. 

I could have used proper squares for this, but I want to save that extra bit of memory by not having lots of references back to the same  grid.
"""
function get_reachable_area(samples_per_axis, mechanics, square::Square, action)
	result = Set()
	for s in possible_outcomes(samples_per_axis, mechanics,  square, action)	
		if !(s ∈ square.grid)
			continue
		end
		
		square′ = box(square.grid, s)
		indices = square′.indices
		if !(indices ∈ result)
			push!(result, indices)
		end
	end
	[result...]
end

"""
	get_barbaric_reachability_function(samples_per_axis, mechanics)

Returns a function `R` of the signature `(square, action) -> squares::Vector{Tuple{Int}}` that computes the set of reachable squares from the initial given square and action. The function returns tuples representing squares instead of members of the Square type for performance reasons.

`samples_per_axis` specifies the number of samples that are taken along the v and p axes. The total number of samples taken will be `samples_per_axis^2`. 
"""
function get_barbaric_reachability_function(samples_per_axis, mechanics)
	return (square, action) ->
		get_reachable_area(samples_per_axis, mechanics, square, action)
end

function draw_barbaric_transition_3D!(samples_per_axis, mechanics::CCMechanics, 
		square::Square, action::CCAction;
		colors=(:black, :gray),
		plotargs...)
	
	samples = [s for s in grid_points(square, samples_per_axis)]
	scatter!([s[1] for s in samples], [s[2] for s in samples], [s[3] for s in samples],
			markersize=2,
			markerstrokewidth=0,
			label="initial square",
			xlabel="v_ego",
			ylabel="v_front",
			color=colors[1],
			zlabel="distance";
			plotargs...)
	
	reach = possible_outcomes(samples_per_axis, mechanics, square, action)
	scatter!([r[1] for r in reach], [r[2] for r in reach], [r[3] for r in reach],
			markersize=2,
			markerstrokewidth=0,
			color=colors[2],
			label="possible outcomes of $action")
end

function draw_barbaric_transition!(samples_per_axis, mechanics::CCMechanics, 
		square::Square, action::CCAction, slice;
		colors=(:black, :gray),
		plotargs...)

	ix, iy = indexof((==(Colon())), slice)
	
	samples = [s for s in grid_points(square, samples_per_axis)]
	scatter!([s[ix] for s in samples], [s[iy] for s in samples],
			markersize=2,
			markerstrokewidth=0,
			label="initial square",
			color=colors[1];
			plotargs...)
	
	reach = possible_outcomes(samples_per_axis, mechanics, square, action)
	scatter!([r[ix] for r in reach], [r[iy] for r in reach],
			markersize=2,
			markerstrokewidth=0,
			color=colors[2],
			label="possible outcomes of $action")
end

"""Update the value of every square reachable from the given `square`.
"""
function set_reachable_area!(samples_per_axis, mechanics, square::Square, action, value)
	
	reachable_area = get_reachable_area(samples_per_axis, mechanics, square::Square, action)
	for indices in reachable_area
		square.grid.array[indices...] = value
	end
end

standard_initialization_function(lower, upper) = lower[3] <= 0 ? 
	0 : actions_to_int(CCAction, instances(CCAction))