is_safe(bounds::Bounds, m::OPMechanics) = is_safe(bounds.lower, m) && is_safe(bounds.upper, m)

function get_op_grid(m::OPMechanics, granularity)
	if granularity isa Number
		granularity = [i == 3 ? 1 : granularity for i in 1:4]
	end
	grid = Grid(granularity, 
		# [t, v, p, l]
		[0, floor(m.v_min - granularity[2]), 0, -granularity[4]], 
		[m.period, ceil(m.v_max + granularity[2]), 2, m.latency + granularity[4]])

	
	initialize!(grid, x -> is_safe(x, m) ?
		actions_to_int([on off]) : actions_to_int([]))
	grid
end

get_randomness_space(m::OPMechanics) = Bounds((-m.fluctuation, -m.fluctuation,), (m.fluctuation, m.fluctuation,))

# Values v and l are unbounded, but we'd like to clamp them to roughly the bounds of the shield.
function clamp_state(grid::Grid, state)
	t, v, p, l = state
	v = clamp(v, grid.bounds.lower[2], grid.bounds.upper[2] - 0.1*grid.granularity[2])
	l = clamp(l, grid.bounds.lower[4], grid.bounds.upper[4] - 0.1*grid.granularity[4])
	t, v, p, l
end

function clamp_state(m::OPMechanics, state)
	ϵ = 0.0001
	t, v, p, l = state
	v = clamp(v, floor(m.v_min - ϵ), ceil(m.v_max + ϵ))
	l = clamp(l, -ϵ, m.latency)
	return t, v, p, l
end

function get_simulation_function(m::OPMechanics)
	simulation_function(state, action, r) = begin
		state′ = simulate_point(m, state, action, r)
		clamp_state(m, state′)
	end
end

# Shield is invalid if the initial state is considered unsafe.
function shield_is_valid(shield)
	square = box(shield, initial_state)
	if get_value(square) == actions_to_int([])
		return false
	end

	return true
end

function shielded(shield, policy)
	return (state) -> begin
		suggested = policy(state)
		state = clamp_state(shield, state)
		partition = box(shield, state)
		allowed = int_to_actions(PumpStatus, get_value(partition))
		if state ∉ shield || length(allowed) == 0 || suggested ∈ allowed
			return suggested
		else
			corrected = rand(allowed)
			return corrected
		end
	end
end