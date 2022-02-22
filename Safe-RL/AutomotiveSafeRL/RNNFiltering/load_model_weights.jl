using Random
using Printf
using Flux
using Flux: truncate!, reset!, batchseq, @epochs, params
using POMDPs
using POMDPModelTools
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutomotiveSensors
using PedCar
using ArgParse
using JLD2
using FileIO
using ProgressMeter
include("RNNFiltering.jl")
using Main.RNNFiltering
using BSON
using BSON: @save, @load
s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
        help = "an integer for the MersenneTwister"
        arg_type = Int64
        default = 1
end
parsed_args = parse_args(ARGS, s)

## RNG SEED
seed = parsed_args["seed"]
rng = MersenneTwister(seed)

# # init continuous state mdp
mdp = PedCarMDP(pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7)
pomdp = UrbanPOMDP(env=mdp.env,
                    sensor = GaussianSensor(false_positive_rate=0.05,
                                            pos_noise = LinearNoise(min_noise=0.5, increase_rate=0.05),
                                            vel_noise = LinearNoise(min_noise=0.5, increase_rate=0.05)),
                   ego_goal = LaneTag(2, 1),
                   max_cars=1,
                   max_peds=1,
                   car_birth=0.05,
                   ped_birth=0.05,
                   obs_dist = ObstacleDistribution(mdp.env, upper_obs_pres_prob=0., left_obs_pres_prob=1.0, right_obs_pres_prob=1.0),
                   max_obstacles=1, # no fixed obstacles
                   lidar=false,
                   ego_start=20,
                   ΔT=0.1)

policy = RandomHoldPolicy(pomdp, 5, 0, UrbanAction(0.), rng);

input_length = n_dims(pomdp)
n_features = 5
output_length = n_features*(pomdp.max_cars + pomdp.max_peds)

model = Chain(LSTM(input_length, 128),
              Dense(128, 64, relu),
              Dense(64, output_length))
@load "weights_$seed.bson" weights
Flux.loadparams!(model, weights)
@save "model_$seed.bson" model

