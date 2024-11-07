function get_transitions(R, actions, grid)
	result = Dict()
	
	for action in instances(actions)
		result[action] = Array{Vector{Any}}(undef, size(grid))
	end
	
	for square in grid
		for action in instances(actions)
			result[action][square.indices...] = R(square, action)
		end
	end
	result
end

function make_shield(R::Function, actions, grid::Grid; max_steps=typemax(Int))
	R_computed = get_transitions(R, actions, grid)
	make_shield(R_computed, actions, grid; max_steps)
end

prebaked_translation_dict = Dict()

function get_translation_dict(actions::Type)
	if haskey(prebaked_translation_dict, actions)
		return prebaked_translation_dict[actions]
	else
		prebaked_translation_dict[actions] = Dict(a => 2^(i-1) for (i, a) in enumerate(instances(actions)))
		return get_translation_dict(actions)
	end
end

# Returns an integer representing the given set of actions
function actions_to_int(actions::Type, list)
	translation_dict = get_translation_dict(actions)
	
	result = 0

	if actions == [] 
		return result
	end
	
	for action in list
		result += translation_dict[action]
	end
	result
end

function get_new_value(R_computed::Dict{Any}, actions, square::Square)
	bad = actions_to_int(actions, []) # No actions allowed in this square; do not go here.
	value = get_value(square)

	if value == bad # Bad squares stay bad. 
		return bad
	end
	
 	result = []

	for action in instances(actions)
		reachable = R_computed[action][square.indices...]
		reachable = [Square(square.grid, i) for i in reachable]
		
		action_allowed = true
		for square′ in reachable
			if get_value(square′) == bad
				action_allowed = false
				break
			end
		end

		if action_allowed 
			push!(result, action)
		end
	end

	actions_to_int(actions, result)
end

#Take a single step in the fixed point compuation.
function shield_step(R_computed::Dict{Any}, actions, grid::Grid)
	grid′ = Grid(grid.G, grid.lower_bounds, grid.upper_bounds)

	for square in grid
		grid′.array[square.indices...] = get_new_value(R_computed, actions, square)
	end
	grid′
end

function make_shield(R_computed::Dict{Any}, actions, grid::Grid; max_steps=typemax(Int))
	i = max_steps
	grid′ = nothing
	while i > 0
		grid′ = shield_step(R_computed, actions, grid)
		if grid′.array == grid.array
			break
		end
		grid = grid′
		i -= 1
	end
	(result=grid′, max_steps_reached=i==0)
end

#Returns an integer representing the given set of actions
function int_to_actions(actions::Type, int::Number)
	translation_dict = get_translation_dict(actions)
	
	result = []
	for (k, v) in translation_dict
		 if int & v != 0
			 push!(result, k)
		 end
	end
	result
end

function draw_shield(shield::Grid, actions; v_ego=0, plotargs...)
	
	square = box(shield, [v_ego, 0, 0])
	index = square.indices[1]
	slice = [index, :, :]

	# Count number of allowed actions in each square
	shield′ = Grid(shield.G, shield.lower_bounds, shield.upper_bounds)
	for square in shield
		square′ = Square(shield′, square.indices)
		
		allowed_actions = get_value(square)
		allowed_actions = int_to_actions(actions, allowed_actions)
		
		set_value!(square′, length(allowed_actions))
	end
	
	draw(shield′, slice, 
		colors=shieldcolors, 
		color_labels=shieldlabels,
		legend=:bottomleft,
		xlabel="v_front",
		ylabel="distance",
		title="v_ego=$v_ego";
		plotargs...)
end

function shielding_function(shield::Grid, actions, fallback_policy::Function, 
		s, action)

	# Return the same action if the state is out of bounds
	if !(s ∈ shield)
		return action
	end
	
	square = box(shield, s)
	allowed = int_to_actions(actions, get_value(square))

	if action ∈ allowed
		return action
	end

	if length(allowed) == 0
		return action
	end

	corrected_action = fallback_policy(s, allowed)

	if !(corrected_action ∈ allowed)
		throw(error("Fallback policy returned illegal action"))
	end

	corrected_action
end


get_shielding_function(shield::Grid, actions, fallback_policy::Function) = 
	(s, a) -> shielding_function(shield, actions, fallback_policy, s, a)
