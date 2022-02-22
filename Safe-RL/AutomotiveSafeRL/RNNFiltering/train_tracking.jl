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
    "--entity"
        help = "choose between car or ped"
        arg_type = String 
        default = "car"
    "--folder"
        help = "location of the training file"
        arg_type = String 
        default = "/scratch/boutonm/"
end
parsed_args = parse_args(ARGS, s)

## RNG SEED 
seed = parsed_args["seed"]
rng = MersenneTwister(seed)
Random.seed!(seed)

## Load training and validation set
folder = parsed_args["folder"]
entity = parsed_args["entity"]
println("Loading existing training data: "*"train_$(entity)_"*string(seed)*".jld2")
train_X, train_Y = load(folder*"train_$(entity)_"*string(seed)*".jld2", "train_X_$(entity)", "train_Y_$(entity)")
println("Loading existing training data: "*"val_$(entity)"*"_"*string(seed)*".jld2")
val_X, val_Y = load(folder*"val_$(entity)_"*string(seed)*".jld2", "val_X_$(entity)", "val_Y_$(entity)")

if parsed_args["resume"] == -1
    model_name = "model_$(entity)_"*string(parsed_args["seed"])
else
    model_name = "model_$(entity)_"*string(parsed_args["resume"])
end
input_length = length(train_X[1][1]) 
output_length = length(train_Y[1][1])

model = Chain(LSTM(input_length, 128),
              Dense(128, 64, relu),
              Dense(64, output_length))
if parsed_args["resume"] != -1
    println("Loading existing model")
    BSON.@load "weights_$(entity)_$(parsed_args["resume"])"*".bson" weights
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
    mask = build_single_presence_mask.(y) # same size as y
    l = mean(mse.(model.(x), y, mask)) # mean over the trajectory
    Flux.truncate!(model)
    Flux.reset!(model)
    return l
end

function mse(ypred, y, mask)
    return sum(mask.*(ypred - y).^2)/length(y)
end

function RNNFiltering.training!(loss, train_data, validation_data, optimizer, n_epochs::Int64)
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
        training_loss /= n_epochs #should be length(train_X)
        println("eval validation loss")
        validation_loss = 0. 
        for d in validation_data
            val_l = loss(d...)
            validation_loss += val_l.data
        end
        validation_loss /= length(validation_data)
        weights = Tracker.data.(params(model))
        BSON.@save model_name*".bson" model
        BSON.@save "weights_$(entity)_$seed"*".bson" weights
        logg = @sprintf("%5d / %5d Train loss %0.3e |  Val loss %1.3e | Grad %2.3e | Epoch time (s) %2.1f | Total time (s) %2.1f",
                                ep, n_epochs, training_loss, validation_loss, grad_norm, epoch_time, total_time)
        println(logg)
        flush(stdout)
    end 
end

optimizer = ADAM(Flux.params(model), 1e-3)

n_epochs = 15
training!(loss, zip(train_X, train_Y), zip(val_X, val_Y), optimizer, n_epochs)

BSON.@save model_name*".bson" model
