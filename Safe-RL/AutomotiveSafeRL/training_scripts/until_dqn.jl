rng = MersenneTwister(1)


using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration
using DeepQLearning, DeepRL
using ProgressMeter, Parameters, JLD

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env = env)

# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

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
                       eval_freq = 10000,
                       arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
                       double_q = true,
                       dueling = true,
                       prioritized_replay = true,
                       verbose = true,
                       logdir = "pedcar-until-log/log2",
                       rng = rng)

policy = solve(solver, mdp)
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")