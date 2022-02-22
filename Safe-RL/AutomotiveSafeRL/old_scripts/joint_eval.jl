rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration, DeepQLearning, DeepRL, LocalApproximationValueIteration
using ProgressMeter, Parameters, JLD
using TensorFlow

include("masking.jl")
include("util.jl")
include("render_helpers.jl")
include("masked_dqn.jl")

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params)

pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.7, 
                   ped_birth=0.7, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.)


### Training


#### Training using DQN in high fidelity environment


# pomdp.action_cost = -0.01
for i=[15, 16, 17, 18, 19]
    max_steps = 500000
    eps_fraction = 0.5 
    eps_end = 0.01 
    solver = DeepQLearningSolver(max_steps = max_steps, eps_fraction = eps_fraction, eps_end = eps_end,
                        lr = 0.0001,                    
                        batch_size = 32,
                        target_update_freq = 5000,
                        max_episode_length = 100,
                        train_start = 40000,
                        buffer_size = 400000,
                        eval_freq = 30000,
                        arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
                        double_q = true,
                        dueling = true,
                        prioritized_replay = true,
                        verbose = true,
                        logdir = "jointmdp-log/log_nm$i",
                        rng = rng)
    env = POMDPEnvironment(pomdp)
    graph=Graph()
    weights_file="jointmdp-log/log_nm$i/weights.jld"
    train_graph = DeepQLearning.build_graph(solver, env, graph)
    policy = DeepQLearning.restore_policy(env, solver, train_graph, weights_file)
    # evaluate resulting policy
    println("\n EVALUATE TRAINED POLICY $i\n")
    @time rewards, steps, violations = evaluation_loop(pomdp, policy, n_ep=10000, max_steps=100, rng=rng);
    print_summary(rewards, steps, violations)
end
