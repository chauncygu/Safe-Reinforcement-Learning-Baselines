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
using BSON: @save
s = ArgParseSettings()
@add_arg_table s begin
    "--seed"
        help = "an integer for the MersenneTwister"
        arg_type = Int64
        default = 1
    "--resume"
        help = "resume training of an existing model"
        arg_type = Int64
        default = -1
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
folder = "/scratch/boutonm/"
max_steps = 400
n_train = 3000
if !isfile(folder*"train_"*string(seed)*".jld2")
    println("Generating $n_train examples of training data")
    train_X, train_Y = collect_set(pomdp, policy, max_steps, rng, n_train)
    save(folder*"train_"*string(seed)*".jld2", "train_X", train_X, "train_Y", train_Y)
else
    println("Loading existing training data: "*"train_"*string(seed)*".jld2")
    JLD2.@load folder*"train_"*string(seed)*".jld2" train_X train_Y
    #train_X, train_Y = train_data[:train_X], train_data[:train_Y]
end
n_val = 500
if !isfile(folder*"val_"*string(seed)*".bson")
    println("Generating $n_val examples of validation data")
    val_X, val_Y = collect_set(pomdp, policy, max_steps, rng, n_val)
    save(folder*"val_"*string(seed)*".jld2", "val_X",val_X, "val_Y", val_Y)
else
    println("Loading existing validation data: "*"val_"*string(seed)*".jld2")
    JLD2.@load folder*"val_"*string(seed)*".jld2" val_X val_Y
    #val_X, val_Y = val_data[:val_X], val_data[:val_Y]
end

if parsed_args["resume"] == -1
    model_name = "model_"*string(parsed_args["seed"])
else
    model_name = "model_"*string(parsed_args["resume"])
end
input_length = n_dims(pomdp) 
n_features = 5
output_length = n_features*(pomdp.max_cars + pomdp.max_peds)

model = Chain(LSTM(input_length, 128),
              Dense(128, 64, relu),
              Dense(64, output_length))
if parsed_args["resume"] != -1
    println("Loading existing model")
    BSON.@load "weights_$(parsed_args["resume"])"*".bson" weights
    Flux.loadparams!(model, weights)
end

macro interrupts(ex)
  :(try $(esc(ex))
    catch e
      e isa InterruptException || rethrow()
      throw(e)
    end)
end

function loss(x, y)
    mask = build_presence_mask.(y) # same size as y
    l = mean(mse.(model.(x), y, mask)) # mean over the trajectory
    Flux.truncate!(model)
    Flux.reset!(model)
    return l
end

function mse(ypred, y, mask)
    return sum(mask.*(ypred - y).^2)/length(y)
end

function training!(loss, train_data, validation_data, optimizer, n_epochs::Int64; logdir::String="log/model/")
    total_time = 0.
    grad_norm = 0.
    println("Starting training")
    for ep in 1:n_epochs 
        epoch_time = @elapsed begin
            opt = Flux.Optimise.runall(optimizer)
            training_loss = 0.
            for d in train_data 
                l = loss(d...)
                @interrupts Flux.back!(l)
                grad_norm = global_norm(params(model))
                opt()
                training_loss += l.data
                flush(stdout) 
            end
        end
        # log 
        total_time += epoch_time
        training_loss /= n_epochs
        println("eval validation loss")
        validation_loss = 0. 
        for d in validation_data
            val_l = loss(d...)
            validation_loss += val_l.data
        end
        validation_loss /= length(validation_data)
        # set_tb_step!(ep)
        # @tb_log training_loss
        # set_tb_step!(ep)
        # @tb_log validation_loss
        # set_tb_step!(ep)
        # @tb_log grad_norm
        weights = Tracker.data.(params(model))
        BSON.@save "weights_$seed"*".bson" weights
        logg = @sprintf("%5d / %5d Train loss %0.3e |  Val loss %1.3e | Grad %2.3e | Epoch time (s) %2.1f | Total time (s) %2.1f",
                                ep, n_epochs, training_loss, validation_loss, grad_norm, epoch_time, total_time)
        println(logg)
        flush(stdout)
    end 
end

optimizer = ADAM(Flux.params(model), 1e-3)

n_epochs = 15
training!(loss, zip(train_X, train_Y), zip(val_X, val_Y), optimizer, n_epochs, logdir="log/"*model_name)

@save model_name*".bson" model
