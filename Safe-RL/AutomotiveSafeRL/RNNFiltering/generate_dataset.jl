using Random
using Printf
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
s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
        help = "an integer for the MersenneTwister"
        arg_type = Int64
        default = 1
    "--folder"
        help = "folder where to save the data"
        arg_type = String 
        default = "/scratch/boutonm/"
    "--ntrain"
        help = "number of training examples"
        arg_type = Int64
        default = 10
    "--nval"
        help = "number of validation examples"
        arg_type = Int64
        default = 10
end
parsed_args = parse_args(ARGS, s)

## RNG SEED 
seed = parsed_args["seed"]
rng = MersenneTwister(seed)
Random.seed!(seed)

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

## Generate data 
folder = parsed_args["folder"]
max_steps = 400
n_train = parsed_args["ntrain"]

println("Generating $n_train examples of training data")
train_X_car, train_Y_car, train_X_ped, train_Y_ped = collect_split_set(pomdp, policy, max_steps, rng, n_train)
@save folder*"train_car_"*string(seed)*".jld2" train_X_car train_Y_car
@save folder*"train_ped_"*string(seed)*".jld2" train_X_ped train_Y_ped

n_val = parsed_args["nval"]
println("Generating $n_val examples of validation data")
val_X_car, val_Y_car, val_X_ped, val_Y_ped = collect_split_set(pomdp, policy, max_steps, rng, n_val)
save(folder*"val_car_"*string(seed)*".jld2", "val_X_car", val_X_car, "val_Y_car", val_Y_car)
save(folder*"val_ped_"*string(seed)*".jld2", "val_X_ped", val_X_ped, "val_Y_ped", val_Y_ped)
