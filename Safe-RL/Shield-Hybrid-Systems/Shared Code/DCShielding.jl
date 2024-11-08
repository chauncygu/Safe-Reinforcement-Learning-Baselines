is_safe(bounds::Bounds, mechanics::DCMechanics) = 		
		is_safe((bounds.lower[1], bounds.lower[2]), mechanics) &&
		is_safe((bounds.upper[1], bounds.upper[2]), mechanics)

function get_dc_grid(m::DCMechanics, granularity)
	if granularity isa Number
		granularity = [granularity, granularity, 1]
	end
	
	grid = Grid(granularity, [
			m.x1_min, 
			m.x2_min - granularity[2],
			m.R_min
		], [
			m.x1_max + granularity[1],
			m.x2_max + granularity[2],
			m.R_max + 1
		])

	initialize!(grid, (b -> is_safe(b, m)))
	return grid
end

get_randomness_space(m::DCMechanics) = Bounds((-m.R_fluctuation,), (m.R_fluctuation,))

# Values v and l are unbounded, but we'd like to clamp them to roughly the bounds of the shield.
function clamp_state(grid::Grid, state::Tuple{Float64, Float64, Float64})
	x1, x2, R = state
	x1 = clamp(x1, grid.bounds.lower[1], grid.bounds.upper[1] - 0.1*grid.granularity[1])
	x2 = clamp(x2, grid.bounds.lower[2], grid.bounds.upper[2] - 0.1*grid.granularity[2])
	R = clamp(R, grid.bounds.lower[3], grid.bounds.upper[3] - 0.1*grid.granularity[3])
	x1, x2, R
end

function clamp_state(m::DCMechanics, state::Tuple{Float64, Float64, Float64})
	ϵ = 0.0001
	x1, x2, R = state
	x1 = clamp(x1, 0, m.x1_max + ϵ)
	x2 = clamp(x2, m.x2_min - ϵ, m.x2_max + ϵ)
	R = clamp(R, m.R_min - ϵ, m.R_max + ϵ)
	x1, x2, R
end

function get_simulation_function(m::DCMechanics)
	return (s, a, r) -> clamp_state(m, simulate_point(m, s, a, r))
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
		allowed = int_to_actions(SwitchStatus, get_value(partition))
		if state ∉ shield || length(allowed) == 0 || suggested ∈ allowed
			return suggested
		else
			corrected = rand(allowed)
			return corrected
		end
	end
end