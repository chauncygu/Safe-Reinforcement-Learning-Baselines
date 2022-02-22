using POMDPs, POMDPToolbox, DiscreteValueIteration, MDPModelChecking
using NearestNeighbors, LocalApproximationValueIteration, LocalFunctionApproximation, StaticArrays
using AutomotiveDrivingModels, AutomotivePOMDPs
using ProgressMeter
using JLD

rng = MersenneTwister(1)

include("masking.jl")
include("util.jl")

function convert_states(mdp::PedCarMDP, sampled_states::Vector{Int64})
    n_routes = 4
    n_features = 4
    nd = n_features*3 + n_routes + 1
    state_space = states(mdp)
    points = Vector{SVector{nd, Float64}}(length(sampled_states))
    for (i, si) in enumerate(sampled_states)
        z = convert_s(Vector{Float64}, state_space[si], mdp)
        points[i] = SVector{nd, Float64}(z...)
    end
    return points
end

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =[VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params)

mdp = PedCarMDP(env=env, pos_res=2.0, vel_res=2.)

println("\n Loading policy...")
flush(STDOUT)
data = JLD.load("pc_lavi_fine100.jld")
println("Policy loaded. \n")
println("Preprocessing loaded policy...")
flush(STDOUT)
sampled_states = data["sampled_states"]
points = convert_states(mdp, sampled_states)
nntree = KDTree(points)
k = 6
knnfa = LocalNNFunctionApproximator(nntree, points, k)
set_all_interpolating_values(knnfa, data["values"])
solver = LocalApproximationValueIterationSolver(knnfa)
policy = LocalApproximationValueIterationPolicy(mdp, solver)
println("Policy ready. \n")

println("Starting evaluation... \n")
flush(STDOUT)
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, policy, n_ep=100, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)
