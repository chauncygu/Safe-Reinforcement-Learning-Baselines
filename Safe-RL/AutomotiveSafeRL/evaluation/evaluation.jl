using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--policy"
        arg_type=String
        default="baseline"
        help = "Choose among 'baseline', 'masked-baseline', 'masked-RL'"
    "--updater"
        arg_type=String 
        default="previous_obs"
        help = "Choose among 'previous_obs', 'tracker'"
    "--scenario"
        arg_type=String 
        default="1-1"
    "--logfile"
        arg_type=String
        default="results.csv"
end
parsed_args = parse_args(ARGS, s)

using Random
using Printf
using Statistics
using StaticArrays
using DataStructures
using DataFrames
using Dates
using CSV

# POMDP and learning
using POMDPs
using BeliefUpdaters
using POMDPPolicies
using POMDPSimulators
using POMDPModelTools
using POMDPModelChecking
using LocalApproximationValueIteration
using DiscreteValueIteration
using RLInterface
using DeepQLearning
using Flux
using BSON
using JLD2
using FileIO

# Driving related Packages
using AutomotiveDrivingModels
using AutomotiveSensors
using AutomotivePOMDPs
using PedCar

# Visualization
using AutoViz
using Reel
using ProgressMeter

include("../src/masking.jl")
include("../src/masked_dqn.jl")
include("../src/qmdp_approximation.jl")
include("../src/decomposed_tracking.jl")
include("../src/decomposition.jl")
include("../src/baseline_policy.jl")
include("../src/util.jl")
include("evaluation_functions.jl")

rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =[VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params)

pomdp = UrbanPOMDP(env=env,
                   sensor = PerfectSensor(),
                  obs_dist = ObstacleDistribution(env, 
                        upper_obs_pres_prob=0., 
                        left_obs_pres_prob=0.0, 
                        right_obs_pres_prob=0.0),
                   ego_goal = LaneTag(2, 1),
                   max_cars=4, 
                   max_peds=4, 
                   car_birth=0., 
                   ped_birth=0., 
                   max_obstacles=1, # no fixed obstacles
                   lidar=false,
                   ego_start=20,
                   ΔT=0.1)
mdp = PedCarMDP(env=env, pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7)
init_transition!(mdp)

# instantiate sub problems
## CAR POMDP FOR TRACKING 1 CAR
car_pomdp = deepcopy(pomdp)
car_pomdp.models = pomdp.models
car_pomdp.max_peds = 0
car_pomdp.max_cars = 1
## PED POMDP FOR TRACKING 1 PEDESTRIAN
ped_pomdp = deepcopy(pomdp)
ped_pomdp.models = pomdp.models
ped_pomdp.max_peds = 1
ped_pomdp.max_cars = 0
## PEDCAR POMDP FOR THE POLICY (Model checking + DQN)
pedcar_pomdp = deepcopy(pomdp)
pedcar_pomdp.models = pomdp.models # shallow copy!
pedcar_pomdp.max_peds = 1
pedcar_pomdp.max_cars = 1
pedcar_pomdp.max_obstacles = 0


# Set sensor characteristics
pomdp.sensor = GaussianSensor(pos_noise = LinearNoise(min_noise=0.5, increase_rate=0.1),
                            vel_noise = LinearNoise(min_noise=0.5, increase_rate=0.1),
                            false_positive_rate = 0.1,
                            false_negative_rate = 0.1)




# Evaluation params 
const N_EPISODES = 1000
const MAX_STEPS = 400


# Set belief updater
updater = nothing
if parsed_args["updater"] == "previous_obs"
    updater = PreviousObservationUpdater()
elseif parsed_args["updater"] == "tracker"
    n_models = 5
    car_models = Vector{Chain}(undef, n_models)
    ped_models = Vector{Chain}(undef, n_models)
    for i=1:n_models
        car_models[i] = BSON.load("../RNNFiltering/model_car_$i.bson")[:model] 
        Flux.loadparams!(car_models[i], BSON.load("../RNNFiltering/weights_car_$i.bson")[:weights])
        ped_models[i] = BSON.load("../RNNFiltering/model_ped_$i.bson")[:model]
        Flux.loadparams!(ped_models[i], BSON.load("../RNNFiltering/weights_ped_$i.bson")[:weights])
    end
    pres_threshold = 0.3
    ref_updaters = Dict(AgentClass.PEDESTRIAN => SingleAgentTracker(ped_pomdp, ped_models, pres_threshold, VehicleDef()),
                    AgentClass.CAR =>  SingleAgentTracker(car_pomdp, car_models, pres_threshold, VehicleDef()))
    updater = MultipleAgentsTracker(pomdp, ref_updaters, Dict{Int64, SingleAgentTracker}())
else 
    throw("Updater $(parsed_args["updater"]) not supported")
end


# Set policy
policy = nothing
if parsed_args["policy"] == "baseline"
    ego_model = get_ego_baseline_model(env)
    policy = EgoBaseline(pomdp, ego_model)
elseif parsed_args["policy"] == "masked-baseline"
    @load "../pc_util_processed_low.jld2" qmat util pol
    safe_policy = ValueIterationPolicy(mdp, qmat, util, pol);
    threshold = 0.99
    mask = SafetyMask(mdp, safe_policy, threshold)
    ego_model = get_ego_baseline_model(env)
    policy = MaskedEgoBaseline(pomdp, pedcar_pomdp, ego_model, mask, UrbanAction[])
elseif parsed_args["policy"] == "masked-RL"
    threshold = 0.99
    @load "../training_scripts/pc_util_processed_low.jld2" qmat util pol
    safe_policy = ValueIterationPolicy(mdp, qmat, util, pol);
    mask = SafetyMask(mdp, safe_policy, threshold)
    qnetwork = BSON.load("../training_scripts/drqn-log/log20/model.bson")[:qnetwork]
    weights = BSON.load("../training_scripts/drqn-log/log20/qnetwork.bson")[:qnetwork]
    Flux.loadparams!(qnetwork, weights)
    dqn_policy = NNPolicy(pedcar_pomdp, qnetwork, actions(pedcar_pomdp), 1)
    masked_policy = MaskedNNPolicy(pedcar_pomdp, dqn_policy, mask)
    policy = DecMaskedPolicy(masked_policy, mask, pedcar_pomdp, (x,y) -> min.(x,y))
elseif parsed_args["policy"] == "RL"
    qnetwork = BSON.load("../training_scripts/rl-log/log3model.bson")[:qnetwork]
    weights = BSON.load("../training_scripts/rl-log/log3qnetwork.bson")[:qnetwork]
    Flux.loadparams!(qnetwork, weights)
    dqn_policy = NNPolicy(pedcar_pomdp, qnetwork, actions(pedcar_pomdp), 1)
    policy = DecPolicy(dqn_policy, pedcar_pomdp, (x,y) -> min.(x,y))
end


# Evaluate

println("Evaluating ", parsed_args["policy"], " policy with ", parsed_args["updater"], " updater")

if parsed_args["scenario"] == "1-1"
    set_static_spec! = set_static_spec_scenario_1_1!
    initialscene_scenario = initialscene_scenario_1_1
elseif parsed_args["scenario"] == "1-2"
    set_static_spec! = set_static_spec_scenario_1_2!
    initialscene_scenario = initialscene_scenario_1_2
elseif parsed_args["scenario"] == "2-1"
    set_static_spec! = set_static_spec_scenario_2_1!
    initialscene_scenario = initialscene_scenario_2_1
elseif parsed_args["scenario"] == "2-2"
    set_static_spec! = set_static_spec_scenario_2_2!
    initialscene_scenario = initialscene_scenario_2_2
elseif parsed_args["scenario"] == "3"
    set_static_spec! = set_static_spec_scenario_3!
    initialscene_scenario = initialscene_scenario_3
end

@time rewards, steps, violations = evaluation_loop(pomdp, policy, updater,
                set_static_spec!, initialscene_scenario,
                n_ep=N_EPISODES, max_steps=MAX_STEPS, rng=rng)
print_summary(rewards, steps, violations)


## Log output
logfile=parsed_args["logfile"]
df = DataFrame(scenario=parsed_args["scenario"], policy=parsed_args["policy"], updater=parsed_args["updater"],n_episodes = N_EPISODES,
               reward_mean = mean(rewards), reward_std = std(rewards),
               steps_mean = mean(steps), steps_std = std(steps),
               violations_mean = mean(violations), violations_std = std(violations), time=Dates.now())

CSV.write(logfile, df, append = isfile(logfile))