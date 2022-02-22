module RNNFiltering

using Flux
using Printf
using Random
using StaticArrays
using ProgressMeter
using POMDPs
using POMDPModelTools
using POMDPSimulators
using BeliefUpdaters
using AutomotiveDrivingModels
using AutomotivePOMDPs
using AutomotiveSensors
using PedCar
using Flux: truncate!, reset!, batchseq, @epochs, params
using BSON: @save

export
    generate_trajectory,
    collect_set,
    generate_split_trajectories,
    collect_split_set, 
    global_norm,
    training!,
    RandomHoldPolicy,
    process_prediction,
    build_presence_mask,
    build_single_presence_mask


function build_presence_mask(y::Vector{Float64}, car_pres_ind=5, ped_pres_ind=10)
    n_features = 5
    mask = ones(length(y))
    mask[car_pres_ind] = 3.
    mask[ped_pres_ind] = 3.
    if y[car_pres_ind] == 0.
        mask[1:car_pres_ind-1] .= 0. 
    end
    if y[ped_pres_ind] == 0.
        mask[ped_pres_ind-n_features+1:ped_pres_ind-1] .= 0.
    end
    return mask
end

function build_single_presence_mask(y::Vector{Float64})
    mask = ones(length(y))
    if y[end] == 0. # car absent 
        mask[1:end-1] .= 0.
    end
    return mask
end

function global_norm(W)
    return maximum(maximum(abs.(w.grad)) for w in W)
end

# function set_tb_step!(t)
#     UniversalTensorBoard.default_logging_session[].global_step =  t
# end

function training!(loss, train_data, opt, n_epochs::Int64; logdir::String="log/model/")
    # set_tb_logdir(logdir)
    total_time = 0.
    grad_norm = 0.
    for ep in 1:n_epochs 
        epoch_time = @elapsed begin
        training_loss = 0.
        for d in train_data 
            l = loss(d...)
            Flux.back!(l)
            grad_norm = global_norm(params(model))
            opt()
            training_loss += l.tracker.data 
        end
        end
        # log 
        total_time += epoch_time
        training_loss /= n_epochs
        validation_loss = mean(loss.(val_X, val_Y)).tracker.data
        # set_tb_step!(ep)
        # @tb_log training_loss
        # set_tb_step!(ep)
        # @tb_log validation_loss
        # set_tb_step!(ep)
        # @tb_log grad_norm
        logg = @sprintf("%5d / %5d Train loss %0.3e |  Val loss %1.3e | Grad %2.3e | Epoch time (s) %2.1f | Total time (s) %2.1f",
                                ep, n_epochs, training_loss, validation_loss, grad_norm, epoch_time, total_time)
        println(logg)
    end 
end

include("data_generation.jl")


end
