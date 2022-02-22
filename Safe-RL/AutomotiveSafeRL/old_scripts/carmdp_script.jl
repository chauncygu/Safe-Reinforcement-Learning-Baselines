rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DeepQLearning, DeepRL
using DiscreteValueIteration
using ProgressMeter, Parameters, JLD

include("util.jl")
include("masking.jl")
include("masked_dqn.jl")

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);
             
mdp = CarMDP(env = env, vel_res=1.0, pos_res=2.0);

### MODEL CHECKING

labels = labeling(mdp);
@printf("\n")
@printf("spatial resolution %2.1f m \n", mdp.pos_res)
@printf("car velocity resolution %2.1f m \n", mdp.vel_res)
@printf("number of states %d \n", n_states(mdp))
@printf("number of actions %d \n", n_actions(mdp))
@printf("\n")

property = "Pmax=? [ (!\"crash\") U \"goal\"]" 
threshold = 0.9999
@printf("Spec: %s \n", property)
@printf("Threshold: %f \n", threshold)

println("Model checking...")
result = model_checking(mdp, labels, property, transition_file_name="car1.tra", labels_file_name="car1.lab")

### MASK
mask = nothing
mask_file = "carmask_new1.jld"
if isfile(mask_file) 
    println("Loading safety mask from carmask.jld")
    mask_data = JLD.load(mask_file)
    mask = SafetyMask(mdp, StormPolicy(mdp, mask_data["risk_vec"], mask_data["risk_mat"]), threshold)
    @printf("Mask threshold %f", mask.threshold)
else
    println("Computing safety mask...")
    mask = SafetyMask(mdp, StormPolicy(mdp, result), threshold)
    JLD.save(mask_file, "risk_vec", mask.policy.risk_vec, "risk_mat", mask.policy.risk_mat)
    println("Mask saved to $mask_file")
end

### EVALUATE MASK 
# rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);

# @time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=1000, max_steps=100, rng=rng);
# print_summary(rewards_mask, steps_mask, violations_mask)


### EVALUATE IN HIGH FIDELITY ENVIRONMENT

pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=0, 
                   car_birth=0.7, 
                   ped_birth=0.3, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.);

# println("EVALUATING IN HIGH FIDELITY ENVIRONMENT")
# rand_pol = RandomMaskedPOMDPPolicy(mask, pomdp, rng);
# @time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, rand_pol, n_ep=1000, max_steps=100, rng=rng);
# print_summary(rewards_mask, steps_mask, violations_mask)

#### Training using DQN

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
                       exploration_policy = masked_linear_epsilon_greedy(max_steps, eps_fraction, eps_end, mask),
                       evaluation_policy = masked_evaluation(mask),
                       verbose = true,
                       logdir = "carmdp-log/log2",
                       rng = rng)

env = POMDPEnvironment(pomdp)
policy = solve(solver, env)
masked_policy = MaskedDQNPolicy(pomdp, policy, mask)


# evaluate resulting policy
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, masked_policy, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

# expected evaluation results:
# Summary for 10000 episodes:
# Average reward: 0.433
# Average # of steps: 19.122
# Average # of violations: 0.000
