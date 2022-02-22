struct SafetyMask{M <: MDP, P <: Policy}
    mdp::M
    policy::P
    threshold::Float64
end

function safe_actions(mask::M, s) where M <: SafetyMask
    A = actiontype(mask.mdp)
    safe_acts = A[]
    vals = actionvalues(mask.policy, s)
    safe = maximum(vals) > mask.threshold ? true : false
    if !safe # follow safe controller
        push!(safe_acts, actions(mask.mdp)[argmax(vals)])
    else
        for (j, a) in enumerate(actions(mask.mdp))
            if vals[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end

# Masked Eps Greedy Policy
"""
Epsilon greedy policy that operates within a safety mask. Both actions from the greedy part and the random part are drawn from the safe actions returned by 
the safety mask.
`MaskedEpsGreedyPolicy{S, A, M}(mdp::MDP{S, A}, epsilon::Float64, mask::M, rng::AbstractRNG)`
"""
struct MaskedEpsGreedyPolicy{M <: SafetyMask, P<:Policy} <: Policy
    val::P # the greedy policy
    epsilon::Float64
    mask::M
    rng::AbstractRNG
end

MaskedEpsGreedyPolicy(mdp::MDP{S, A}, epsilon::Float64, mask::M, rng::AbstractRNG) where {S, A, M <: SafetyMask} = MaskedEpsGreedyPolicy{M}(ValueIterationPolicy(mdp), epsilon, mask, rng)

function POMDPs.action(policy::MaskedEpsGreedyPolicy{M}, s) where M
    acts = safe_actions(policy.mask, s)
    if rand(policy.rng) < policy.epsilon
        return rand(policy.rng, acts)
    else
        return best_action(acts, policy.val, s)
    end
end

"""
A value policy that operates within a safety mask, it takes the action in the set of safe_actions that maximizes the given value function. 
`MaskedValuePolicy{M <: SafetyMask}(val::ValuePolicy, mask::M`
"""
struct MaskedValuePolicy{M <: SafetyMask, P<:Policy} <: Policy
    val::P
    mask::M
end

function POMDPs.action(policy::MaskedValuePolicy{M}, s) where M
    acts = safe_actions(policy.mask, s)
    return best_action(acts, policy.val, s)
end

function best_action(acts::Vector{A}, policy::P, s) where {A,P<:Policy}
    si = stateindex(policy.mdp, s)
    best_a = first(acts)
    best_val = -Inf
    for a in acts
        val = value(policy, s, a)
        if val > best_val
            best_a = a 
            best_val = val 
        end
    end
    return best_a
end

## Problem specific methods

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return safe_actions(pomdp, mask, s, CAR_ID)
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return safe_actions(mask, o[:], CAR_ID)
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{CarMDP, P}, s::UrbanState, car_id::Int64) where P <: Policy
    s_mdp = get_mdp_state(mask.mdp, pomdp, s, car_id)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    # si = stateindex(mdp, itp_states[argmax(itp_weights)])
    # p_sa = mask.risk_mat[si, :]
#     p_sa_itp = zeros(length(itp_states), n_actions(mask.mdp))
#     for (i, ss) in enumerate(itp_states)
#         si = stateindex(mask.mdp, ss)
#         p_sa_itp[i, :] += itp_weights[i]*mask.risk_mat[si,:]
#     end
#     p_sa = minimum(p_sa_itp, 1)
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        vals = actionvalues(mask.policy, ss)
        p_sa += itp_weights[i]*vals
    end
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    action_space = actions(mask.mdp)
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, action_space[argmax(p_sa)])
    else
        for (j, a) in enumerate(action_space)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    # println("coucou ")
    # global debug_i
    # println("Safe acts $([a.acc for a in safe_acts])")
    # println(" i ", debug_i)
    # debug_i += 1
    return safe_acts
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return safe_actions(mask, s, PED_ID)
end

function safe_actions(mask::SafetyMask{PedMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return safe_actions(mask, o[:])
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedMDP, P},s::UrbanState) where P <: Policy   
    return safe_actions(mask, s, PED_ID)
end

function safe_actions(mask::SafetyMask{PedMDP, P}, s::UrbanState, ped_id) where P <: Policy
    s_mdp = get_mdp_state(mask.mdp, s, ped_id)
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        vals = actionvalues(mask.policy, ss)
        p_sa += itp_weights[i]*vals
    end
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    action_space = actions(mask.mdp)
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, action_space[argmax(p_sa)])
    else
        for (j, a) in enumerate(action_space)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return safe_actions(pomdp, mask, s, PED_ID, CAR_ID)
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return safe_actions(mask, o[:], PED_ID, CAR_ID)
end

function safe_actions(mask::SafetyMask{M, LocalApproximationValueIterationPolicy}, o::Array{Float64}) where M <: Union{PedMDP, PedCarMDP}
    s = convert_s(state_type(mask.mdp), o, mask.mdp)
    return safe_actions(mask, s)
end

function POMDPPolicies.actionvalues(policy::LocalApproximationValueIterationPolicy, s::S) where S <: Union{PedMDPState, PedCarMDPState}
    if !s.crash && isterminal(policy.mdp, s)
        return ones(n_actions(policy.mdp))
    else
        q = zeros(n_actions(policy.mdp))
        for i = 1:n_actions(policy.mdp)
            q[i] = action_value(policy, s, policy.action_map[i])
        end
        return q
    end
end

function safe_actions(mask::SafetyMask, p_sa::Vector{Float64})
    safe_acts = UrbanAction[]
    sizehint!(safe_acts, n_actions(mask.mdp))
    action_space = actions(mask.mdp)
    if maximum(p_sa) <= mask.threshold
        push!(safe_acts, action_space[argmax(p_sa)])
    else
        for (j, a) in enumerate(action_space)
            if p_sa[j] > mask.threshold
                push!(safe_acts, a)
            end
        end
    end
    return safe_acts
end


function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, s::UrbanState, ped_id, car_id) where P <: Policy
    p_sa = compute_probas(pomdp, mask, s, ped_id, car_id)
    return safe_actions(mask, p_sa)
end

function compute_probas(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, o::UrbanObs) where P <: Policy
    s = obs_to_scene(pomdp, o)
    return compute_probas(pomdp, mask, s, PED_ID, CAR_ID)
end

function compute_probas(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, o::Array{Float64, 2}) where P <: Policy
    d, dd = size(o)
    @assert dd == 1
    return compute_probas(mask, o[:], PED_ID, CAR_ID)
end

function compute_probas(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, s::UrbanState, ped_id, car_id) where P <: Policy
    s_mdp = PedCar.get_mdp_state(mask.mdp, pomdp, s, ped_id, car_id)
    # println("route s_mdp : ", s_mdp.route)
    # println("projected pedestrian : ", s_mdp.ped)
    # println("original pedestrian : ", s[findfirst(ped_id, s)])
    itp_states, itp_weights = interpolate_state(mask.mdp, s_mdp)
    # compute risk vector
    p_sa = zeros(n_actions(mask.mdp))
    for (i, ss) in enumerate(itp_states)
        # println("interpolated ped: ", ss.ped)
        vals = actionvalues(mask.policy, ss)
        p_sa += itp_weights[i]*vals
    end
    return p_sa
end

function POMDPModelTools.action_info(policy::MaskedEpsGreedyPolicy{M}, s) where M <: SafetyMask
    return action(policy, s), [safe_actions(policy.mask, s), s]
end

# ## new policy type to work with UrbanPOMDP

struct RandomMaskedPOMDPPolicy{M} <: Policy 
    mask::M
    pomdp::UrbanPOMDP
    rng::AbstractRNG
end

struct SafePOMDPPolicy{M} <: Policy
    mask::M 
    pomdp::UrbanPOMDP
end

function POMDPs.action(policy::RandomMaskedPOMDPPolicy, s)
    acts = safe_actions(policy.pomdp, policy.mask, s)
    if isempty(acts)
        def_a = UrbanAction(-4.0)
        # warn("WARNING: No feasible action at this step, choosing default action $(def_a.acc)m/s^2")
        return def_a
    end
    return rand(policy.rng, acts)
end

function POMDPModelTools.action_info(policy::RandomMaskedPOMDPPolicy{M}, s) where M
    sa = safe_actions(policy.pomdp, policy.mask, s)
    probas = compute_probas(policy.pomdp, policy.mask, s)
    ss = obs_to_scene(policy.pomdp, s)
    route = get_mdp_state(policy.mask.mdp, policy.pomdp, ss, PED_ID, CAR_ID).route
    return action(policy, s), (sa, probas, route)
end

function POMDPs.action(policy::SafePOMDPPolicy{M}, s) where M 
    probas = compute_probas(policy.pomdp, policy.mask, s)
    ai = argmax(probas)
    return actions(policy.pomdp)[ai]
end

function POMDPModelTools.action_info(policy::SafePOMDPPolicy{M}, s) where M 
    sa = safe_actions(policy.pomdp, policy.mask, s)
    probas = compute_probas(policy.pomdp, policy.mask, s)
    ss = obs_to_scene(policy.pomdp, s)
    route = get_mdp_state(policy.mask.mdp, policy.pomdp, ss, PED_ID, CAR_ID).route
    return action(policy, s), (sa, probas, route)
end


struct JointMask{P <: MDP, M <: SafetyMask, I}
    problems::Vector{P}
    masks::Vector{M}
    ids::Vector{I}
end

function safe_actions(pomdp::UrbanPOMDP, mask::JointMask, s::S) where S
    acts = intersect([safe_actions(pomdp, m, s) for m in mask.masks]...) 
    if isempty(acts)
        return UrbanAction[UrbanAction(-4.0)]
    end
    return acts       
end
