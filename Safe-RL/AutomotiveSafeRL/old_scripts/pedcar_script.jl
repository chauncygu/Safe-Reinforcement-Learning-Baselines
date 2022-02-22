rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DeepQLearning, DeepRL
using DiscreteValueIteration, LocalApproximationValueIteration
using ProgressMeter, Parameters, JLD

include("util.jl")
include("masking.jl")
include("masked_dqn.jl")
params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params)

mdp = PedCarMDP(env=env, pos_res=2.0, vel_res=2.);

state_space = states(mdp);

vi_data = JLD.load("pc_util_f.jld")
@showprogress for s in state_space
    if !s.crash && isterminal(mdp, s)
        si = stateindex(mdp, s)
        vi_data["util"][si] = 1.0
        vi_data["qmat"][si, :] = ones(n_actions(mdp))
    end
end
policy = ValueIterationPolicy(mdp, vi_data["qmat"], vi_data["util"], vi_data["pol"]);
threshold = 0.99
mask = SafetyMask(mdp, policy, threshold)


pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.3, 
                   ped_birth=0.3, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.,
                   ego_start=20);
rand_pol = RandomMaskedPOMDPPolicy(mask, pomdp, rng);
println("\nEvaluation in continuous environment: \n")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, rand_pol, n_ep=1000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)
flush(STDOUT)

pomdp.action_cost = -0.01
max_steps = 500000
eps_fraction = 0.5 
eps_end = 0.01 
solver = DeepQLearningSolver(max_steps = max_steps, eps_fraction = eps_fraction, eps_end = eps_end,
                       lr = 0.0001,                    
                       batch_size = 32,
                       target_update_freq = 5000,
                       max_episode_length = 200,
                       train_start = 40000,
                       buffer_size = 400000,
                       eval_freq = 30000,
                       arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
                       double_q = true,
                       dueling = true,
                       prioritized_replay = true,
                       exploration_policy = masked_linear_epsilon_greedy(max_steps, eps_fraction, eps_end, mask),
                       evaluation_policy = masked_evaluation(mask),
                       verbose = true,
                       logdir = "joint-log/log8",
                       rng = rng)

env = POMDPEnvironment(pomdp)
policy = solve(solver, env)
masked_policy = MaskedDQNPolicy(pomdp, policy, mask)
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")

# evaluate resulting policy
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, masked_policy, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)
flush(STDOUT)
