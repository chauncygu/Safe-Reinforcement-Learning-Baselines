using POMDPs, POMDPToolbox, MDPModelChecking
using AutomotivePOMDPs, AutomotiveDrivingModels
using DeepQLearning, DeepRL


rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env = env, pos_res=1., vel_res=1., ped_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))
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

rng = MersenneTwister(1)

# reachability analysis
pomdp.collision_cost = 0.
pomdp.γ = 1.
pomdp.goal_reward = 1.

max_steps = 1000000
eps_fraction = 0.9 
eps_end = 0.01 
solver = DeepQLearningSolver(max_steps = max_steps, eps_fraction = eps_fraction, eps_end = eps_end,
                       lr = 0.0001,                    
                       batch_size = 32,
                       target_update_freq = 7000,
                       max_episode_length = 50,
                       train_start = 40000,
                       buffer_size = 400000,
                       eval_freq = 30000,
                       arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
                       double_q = true,
                       dueling = true,
                       prioritized_replay = true,
                       verbose = true,
                       logdir = "mc-log/log4",
                       rng = rng)

env = POMDPEnvironment(pomdp)
policy = solve(solver, env)
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")
# evaluate resulting policy
println("\n EVALUATE TRAINED POLICY \n")
@time rewards, steps, violations = evaluation_loop(pomdp, policy, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards, steps, violations)

# function MDPModelChecking.labels(mdp::PedCarMDP, s::PedCarMDPState)
#     if crash(mdp, s)
#         return ["crash"]
#     elseif s.ego.posF.s >= get_end(mdp.env.roadway[mdp.ego_goal]) &&
#             get_lane(mdp.env.roadway, s.ego).tag == mdp.ego_goal
#         return ["goal"]
#     else
#         return ["!crash", "!good"]
#     end
# end


# property = "!crash U goal"
# mdp = PedCarMDP()

# max_steps = 500000
# eps_fraction = 0.5 
# eps_end = 0.01 
# dqn_solver = DeepQLearningSolver(max_steps = max_steps, eps_fraction = eps_fraction, eps_end = eps_end,
#                        lr = 0.0001,                    
#                        batch_size = 32,
#                        target_update_freq = 5000,
#                        max_episode_length = 100,
#                        train_start = 40000,
#                        buffer_size = 400000,
#                        eval_freq = 30000,
#                        arch = QNetworkArchitecture(conv=[], fc=[32,32,32]),
#                        double_q = true,
#                        dueling = true,
#                        prioritized_replay = true,
#                        verbose = true,
#                        logdir = "pedcar-log/log1",
#                        rng = rng)

# solver = ModelCheckingSolver(property=property,
#                              solver=dqn_solver)
# policy = solve(solver, mdp)

# # save DQN results
# DeepQLearning.save(dqn_solver, policy.policy, weights_file=dqn_solver.logdir*"/weights.jld", problem_file=dqn_solver.logdir*"/problem.jld")
