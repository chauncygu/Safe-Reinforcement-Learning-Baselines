rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration, DeepQLearning, DeepRL
using ProgressMeter, Parameters, JLD
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--log"
        arg_type=String
        default="log"
    "--goal"
        arg_type=Float64
        default=1.0
end
parsed_args = parse_args(ARGS, s)

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

ped_mdp = PedMDP(env = env, vel_res=1., pos_res=1., ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0), ped_birth=0.7)
car_mdp = CarMDP(env = env, vel_res=1., pos_res=2.)

threshold = 0.9999
ped_mask_file = "pedmask_new1.jld"
car_mask_file = "carmask_new1.jld"
ped_mask_data = load(ped_mask_file)
car_mask_data = load(car_mask_file)
ped_mask = SafetyMask(ped_mdp, StormPolicy(ped_mdp, ped_mask_data["risk_vec"], ped_mask_data["risk_mat"]), threshold)
car_mask = SafetyMask(car_mdp, StormPolicy(car_mdp, car_mask_data["risk_vec"], car_mask_data["risk_mat"]), threshold);

pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=1, 
                   car_birth=0.3, 
                   ped_birth=0.3, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.)
pomdp.goal_reward = parsed_args["goal"]
pomdp.action_cost = -0.01
masks = SafetyMask[ped_mask, car_mask]
ids = [101, 2]
joint_mask = JointMask([ped_mdp, car_mdp], masks, ids)
rand_pol = RandomMaskedPOMDPPolicy(joint_mask, pomdp, rng)

# @time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, rand_pol, n_ep=10000, max_steps=100, rng=rng);
# print_summary(rewards_mask, steps_mask, violations_mask)

### Training


#### Training using DQN in high fidelity environment



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
                       exploration_policy = masked_linear_epsilon_greedy(max_steps, eps_fraction, eps_end, joint_mask),
                       evaluation_policy = masked_evaluation(joint_mask),
                       verbose = true,
                       logdir = "jointmdp-log/"*parsed_args["log"],
                       rng = rng)

pomdp.action_cost = -0.01
env = POMDPEnvironment(pomdp)
policy = solve(solver, env)
DeepQLearning.save(solver, policy, weights_file=solver.logdir*"/weights.jld", problem_file=solver.logdir*"/problem.jld")
masked_policy = MaskedDQNPolicy(pomdp, policy, joint_mask)

# evaluate resulting policy
println("\n EVALUATING TRAINED POLICY WITH SAFETY MASK \n")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, masked_policy, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

# Summary for 10000 episodes:
# Average reward: 0.290
# Average # of steps: 34.186
# Average # of violations: 0.000
