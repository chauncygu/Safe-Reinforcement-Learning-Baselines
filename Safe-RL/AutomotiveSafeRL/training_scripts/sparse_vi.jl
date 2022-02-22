using Revise
using Printf
using Random
using POMDPs
using POMDPModelTools
using POMDPPolicies
using BeliefUpdaters
using DiscreteValueIteration
using POMDPSimulators
using AutomotivePOMDPs, AutomotiveDrivingModels
using BSON, StaticArrays
using JLD2
using FileIO
using PedCar
using LocalApproximationValueIteration
using ProgressMeter

rng = MersenneTwister(1)

include(joinpath(@__DIR__, "../src/masking.jl"))
include(joinpath(@__DIR__, "../src/util.jl"))


params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCar.PedCarMDP(env=env, pos_res=3.0, vel_res=2.0, ped_birth=0.7, car_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))
init_transition!(mdp)
# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

solver = SparseValueIterationSolver(max_iterations=20, belres=1e-4, include_Q=true, verbose=true)
policy_file = "pedcar_utility.jld2"
#policy = ValueIterationPolicy(mdp, include_Q=true)
if isfile(policy_file)
  println("Loading policy file $policy_file")
  JLD2.@load policy_file qmat util pol 
  solver.init_util = util
end
policy = solve(solver, mdp)

save(policy_file, "util", policy.util, "qmat", policy.qmat, "pol",policy.policy)
#bson(policy_file, util=policy.util, qmat=policy.qmat, pol=policy.policy)
#bson("pedcar_policy.bson", policy=policy)

threshold = 0.999
mdp.collision_cost = -1.0
mask = SafetyMask(mdp, policy, threshold);
rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);
println("Evaluation in discretized environment: \n ")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)
mdp.collision_cost = 0.

solver.init_util = policy.util
policy = solve(solver, mdp) # resume


threshold = 0.999
mdp.collision_cost = -1.0
mask = SafetyMask(mdp, policy, threshold);
rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);
println("Evaluation in discretized environment: \n ")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=10000, max_steps=100, rng=rng);print_summary(rewards_mask, steps_mask, violations_mask)
mdp.collision_cost = 0.0

save(policy_file, "util", policy.util, "qmat", policy.qmat, "pol",policy.policy)
#bson(policy_file, util=policy.util, qmat=policy.qmat, pol=policy.policy)
#bson("pedcar_policy.bson", policy=policy)
