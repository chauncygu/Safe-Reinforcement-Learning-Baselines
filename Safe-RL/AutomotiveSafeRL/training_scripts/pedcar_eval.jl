using Printf
using Random
using POMDPs
using POMDPModelTools
using DiscreteValueIteration
using POMDPSimulators
using AutomotiveDrivingModels
using AutomotivePOMDPs
using StaticArrays
using JLD2
using FileIO
using PedCar
using MDPModelChecking
using LocalApproximationValueIteration
using ProgressMeter

include("../src/masking.jl")
include("../src/util.jl")

rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =[VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCar.PedCarMDP(env=env, pos_res=2.0, vel_res=2.0, ped_birth=0.7, car_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))
init_transition!(mdp)

@load "pc_util_processed_low.jld2" qmat util pol
safe_policy = ValueIterationPolicy(mdp, qmat, util, pol)
threshold = 0.99
mask = SafetyMask(mdp, safe_policy, threshold);
discrete_safe_random = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng)

println("Evaluating safe policy")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, safe_policy, n_ep=10000, max_steps=400, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

println("Evaluating safe random with threshold $threshold")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, discrete_safe_random, n_ep=10000, max_steps=400, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)
