using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration
using ProgressMeter, Parameters, JLD
using LocalFunctionApproximation, LocalApproximationValueIteration
using NearestNeighbors, StatsBase

function sample_points(mdp::PedCarMDP, n_samples::Int64, rng::AbstractRNG)
    # sample points for the approximator
    n_routes = 4
    n_features = 4
    nd = n_features*3 + n_routes + 1 
    sampled_states = sample(rng, 1:n_states(mdp), n_samples, replace=false)
    points = Vector{SVector{nd, Float64}}(n_samples)
    for i=1:length(sampled_states)
        s = ind2state(mdp, sampled_states[i])
        z = convert_s(Vector{Float64}, s, mdp)
        points[i] = SVector{nd, Float64}(z...)
    end
    return points, sampled_states
end

function convert_states(mdp::PedCarMDP, sampled_states::Vector{Int64})
    n_routes = 4
    n_features = 4
    nd = n_features*3 + n_routes + 1
    points = Vector{SVector{nd, Float64}}(length(sampled_states))
    for (i, si) in enumerate(sampled_states)
        s = ind2state(mdp, si)
        z = convert_s(Vector{Float64}, s, mdp)
        points[i] = SVector{nd, Float64}(z...)
    end
    return points
end

function set_terminal_costs!(mdp::PedCarMDP, knnfa::LocalNNFunctionApproximator)
    for (i, pt) in enumerate(knnfa.nnpoints)
        s = convert_s(PedCarMDPState, pt, mdp)
        if !s.crash && isterminal(mdp, s) 
            knnfa.nnvalues[i] = 1.0
        elseif s.crash 
            knnfa.nnvalues[i] = 0.0
        end
    end
end

rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env = env, pos_res=1., vel_res=1., ped_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))

# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

N_SAMPLES = 100000
k = 20
knnfa = nothing
sampled_states = nothing
policy_file = "lavi_20neighb100k.jld"
if isfile(policy_file)
    data = load(policy_file)
    sampled_states = data["sampled_states"]
    points = convert_states(mdp, sampled_states)
    nntree = KDTree(points)
    knnfa = LocalNNFunctionApproximator(nntree, points, k)
    set_all_interpolating_values(knnfa, data["values"])
else
    points, sampled_states = sample_points(mdp, N_SAMPLES, rng)
    nntree = KDTree(points)
    knnfa = LocalNNFunctionApproximator(nntree, points, k)
    #set_terminal_costs!(mdp, knnfa)
end

approx_solver = LocalApproximationValueIterationSolver(knnfa, verbose=true, max_iterations=40, is_mdp_generative=false)
policy = solve(approx_solver, mdp)

JLD.save(policy_file, "sampled_states", sampled_states, "values", policy.interp.nnvalues ) 
