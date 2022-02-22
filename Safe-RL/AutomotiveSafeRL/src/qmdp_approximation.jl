## Define Belief Updater for RNN Observation
# the belief is represented by a vector 
# 

struct PedCarRNNUpdater <: Updater 
    models::Vector{Chain}
    mdp::PedCarMDP
    pomdp::UrbanPOMDP
end

struct PedCarRNNBelief
    predictions::Vector{Vector{Float64}}
    obs::Vector{Float64}
end


function POMDPs.update(up::PedCarRNNUpdater, bold::PedCarRNNBelief, a, o::Vector{Float64})
    predictions = predict!(up.models, o)
    return PedCarRNNBelief(predictions, o)
end    

function predict!(models::Vector{Chain}, o::Vector{Float64})
    n_models = length(models)
    predictions = Vector{Vector{Float64}}(undef, n_models)
    for i=1:n_models       
        pred = models[i](o).data
        predictions[i] = pred
    end
    return predictions
end

function reset_updater!(up::PedCarRNNUpdater)
    for m in up.models
        Flux.reset!(m)
    end
end

function POMDPs.action(policy::MaskedNNPolicy, b::PedCarRNNBelief)
    safe_acts = safe_actions(policy.problem, policy.mask, b)
    val = actionvalues(policy.q, b)
    act = best_action(safe_acts, val, policy.problem)
    return act
end

function POMDPPolicies.actionvalues(policy::MaskedNNPolicy, b::PedCarRNNBelief)
    n_features = 4
    pomdp = policy.problem
    vals = zeros(length(actions(pomdp)))
    for i=1:length(b.predictions)
        # bb, _ = RNNFiltering.process_prediction(pomdp, b.predictions[i], b.obs)
        vals += actionvalues(policy.q, b.predictions[i])[:]
    end
    return vals./length(b.predictions)
end

function POMDPPolicies.actionvalues(policy::NNPolicy{UrbanPOMDP}, b::PedCarRNNBelief)
    n_features = 4
    pomdp = policy.problem
    vals = zeros(length(actions(pomdp)))
    for i=1:length(b.predictions)
        # bb, _ = RNNFiltering.process_prediction(pomdp, b.predictions[i], b.obs)
        vals += actionvalues(policy, b.predictions[i])[:]
    end
    return vals./length(b.predictions)
end

function POMDPPolicies.action(policy::NNPolicy, b::PedCarRNNBelief)
    vals = actionvalues(policy, b)
    ai = argmax(vals)
    return actions(policy.problem)[ai]
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, b::PedCarRNNBelief, ped_id::Int64=PED_ID, car_id::Int64=CAR_ID) where P <: Policy
    vals = compute_probas(pomdp, mask, b, ped_id, car_id)
    
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    action_space = actions(mask.mdp)
    if maximum(vals) <= mask.threshold
        push!(safe_acts, action_space[argmax(vals)])
    else
        for (j, a) in enumerate(action_space)
            if vals[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end

function compute_probas(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, b::PedCarRNNBelief, ped_id::Int64=PED_ID, car_id::Int64=CAR_ID) where P <: Policy
    vals = zeros(n_actions(pomdp))
    for i=1:length(b.predictions)
        # bb, _ = RNNFiltering.process_prediction(pomdp, b.predictions[i], b.obs)
        s = obs_to_scene(pomdp, b.predictions[i])
        # println(compute_probas(pomdp, mask, s, ped_id, car_id))
        vals += compute_probas(pomdp, mask, s, ped_id, car_id)# need to change b
    end
    vals ./= length(b.predictions)
    return vals
end

# function AutomotivePOMDPs.obs_to_scene(pomdp::UrbanPOMDP, b::PedCarRNNBelief)
#     return obs_to_scene(pomdp, b.obs)
# end

## Perfect Updater no obstacles 
struct PerfectSensorUpdater <: Updater
    pomdp::UrbanPOMDP
end

function POMDPs.update(bu::PerfectSensorUpdater, old_b::Vector{Float64}, action, obs::Vector{Float64}) 
    if pomdp.max_obstacles != 0. 
        return obs[1:end - pomdp.n_features*pomdp.max_obstacles]
    else
        return obs 
    end
end
