# Include along with Squares.jl, and package ReachabilityAnalyis

"""
	get_rigorous_reachability_function(mechanics, [algorithm])

Returns a function `R` of the signature `square, action -> squares::Vector{Tuple{Int}}` that computes the set of reachable squares from the initial given square and action. The function returns tuples representing squares instead of members of the Square type for performance reasons.

`algorithm` could be one of:

 - BOX δ=0.002 (20ms/action)
 - BOX δ=0.01 (4ms/action)
 - GLGM06 δ=0.002 (10ms/action)
 - LGG09 δ=0.005 (6ms/action, sometimes incorrect)
"""
function get_rigorous_reachability_function(mechanics, algorithm=nothing)
	H = create_bouncing_ball_hybrid_system()

	algorithm = something(algorithm, BOX(δ=0.02))
	return (square, action) -> 
		get_reachable_squares(H, mechanics, square, action; algorithm)
end

# Variables inside the matrices cause errors, so I had to hardcode these values.
function create_bouncing_ball_hybrid_system()
    # symbolic variables
    var = @variables v p

    # falling mode with invariant p >= 0
    A = [0 0.; 1 0]
    b = [-9.81, 0] # HARDCODED: g
    freefall = @system(z' = A*z + b, z ∈ HalfSpace(p ≥ 0, var))

    # guard p ≤ 0 && v ≤ 0
    floor = HPolyhedron([p ≤ 0, v ≤ 0], var)

    # reset map v⁺ := -cv
    impact = ConstrainedLinearMap([ -0.85 0; 0 1], floor) # HARDCODED: β2 - ε2

    # hybrid system
    H = HybridSystem(freefall, impact)

    return H
end

function get_hyperrectangle(mechanics, square, action)::Hyperrectangle
	t_hit, g, β1, ϵ1, β2, ϵ2, v_hit, p_hit  = mechanics
	Ivl, Ivu, Ipl, Ipu = bounds(square)
	
	if action != "hit"
		return Hyperrectangle(low=[Ivl, Ipl], high=[Ivu, Ipu])
	end

	# Hitting the ball is an atomic action at the very start of the timestep.
	# Therefore, a hit can be modelled simply by updating the bounds.
	# The bounds may not match up to the grid after the update, but this does not matter to the initial value problem.
	if !(Ipu <= p_hit || Ivu <= v_hit)
		if Ivl < 0
			if Ivu <= 0
				Ivl, Ivu = v_hit, v_hit
			else
				@warn "Doing a huge overapproximation. This is due to an edge case caused by an unusual grid layout."
				Ivl, Ivu = -(β2 - ϵ2)*Ivu + v_hit, v_hit
			end
		else
			Ivl, Ivu = -(β2 - ϵ2)*Ivu + v_hit, -(β2 - ϵ2)*Ivl + v_hit
		end
	end
		
	Hyperrectangle(low=[Ivl, Ipl], high=[Ivu, Ipu])
end

# Get the bounds from a 2D hyperrectangle
function hrec_bounds(hrec::Hyperrectangle)
	Ivl, Ivu, Ipl, Ipu = 
					hrec.center[1] - hrec.radius[1], 
					hrec.center[1] + hrec.radius[1],
					hrec.center[2] - hrec.radius[2], 
					hrec.center[2] + hrec.radius[2] 
end

function square_indices_from_reach_sets(grid::Grid, reach_sets::Vector{ReachSet{T, S}}) where T where S<:LazySet
	result = Set()
	for reach_set in reach_sets
		hyperrectangle = overapproximate(reach_set.X)
		bounds = hrec_bounds(hyperrectangle)
		square_indices = cover(grid, bounds...)
		union!(result, square_indices)
	end
	[result...] # Expects a vector for some reason
end

function get_reachable_squares(H, mechanics, square, action; algorithm=nothing)
	t_hit, g, β1, ϵ1, β2, ϵ2, v_hit, p_hit  = mechanics
	time_step = t_hit
	
	# Initial approximate position. 
	# The hit action will change the initial velocity bounds, if applicable.
	Z0 = get_hyperrectangle(mechanics, square, action)

	## Black magic ##
	# Initiate an automaton
	# Define an Initial Value Problem
	prob = @ivp(H, z(0) ∈ Z0);	
	# Unknown. The number of directions has to match dimensionality
	boxdirs = BoxDirections{Float64, Vector{Float64}}(2)
	Thull = TemplateHullIntersection(boxdirs)
	# Get solution to the initial value problem up to time step T 
	sol = solve(prob, T=time_step,
			alg=algorithm,
            #alg=GLGM06(δ=0.01), # Errors out. 
			#alg=LGG09(δ=δ, template=boxdirs), # This one works
            clustering_method=LazyClustering(1, convex=false),
            intersection_source_invariant_method=Thull,
            intersection_method=Thull)
	# The solution covers the entire trace throughout the time step.
	# Extract just the approximate location at the end of the time step.
	Z1 = sol(time_step)

	# Function to turn hyper rectangles into squares
	square_indices_from_reach_sets(square.grid, Z1)
end

struct AlgorithmInfo
	algorithm::ReachabilityAnalysis.AbstractContinuousPost
	milliseconds_per_call::Number
	label::AbstractString
end