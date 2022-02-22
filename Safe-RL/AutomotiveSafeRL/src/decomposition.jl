# utility decomposition: 

struct DecMaskedPolicy{A <: Policy, M <: SafetyMask, P <: Union{MDP, POMDP}} <: Policy
    policy::A
    mask::M
    problem::P
    op # reduction operator
end

function POMDPPolicies.actionvalues(policy::DecMaskedPolicy, dec_belief::Dict)  # no hidden state!
    return reduce(policy.op, actionvalues(policy.policy, b) for (_,b) in dec_belief)
end

function POMDPs.action(p::DecMaskedPolicy, b::Dict)
    if isempty(b)
        return UrbanAction(2.0)
    else
        safe_acts, _ = safe_actions(p.problem, p.mask, b)
        val = actionvalues(p, b)
        act = best_action(safe_acts, val, p.problem)
    end
    return act
end

struct DecPolicy{A <: Policy, P <: Union{MDP, POMDP}} <: Policy 
    policy::A
    problem::P 
    op # reduction operator 
end

function POMDPPolicies.actionvalues(policy::DecPolicy, dec_belief::Dict)  # no hidden state!
    return reduce(policy.op, actionvalues(policy.policy, b) for (_,b) in dec_belief)
end

function POMDPPolicies.action(p::DecPolicy, b::Dict)
    if isempty(b)
        return UrbanAction(2.0)
    else
        vals = actionvalues(p, b)
        ai = argmax(vals)
        return actions(p.problem)[ai]
    end   
end

function safe_actions(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, b::Dict{I, PedCarRNNBelief}) where {P <: Policy,I}
    safe_acts = Dict{Tuple{Int64, Int64}, Vector{UrbanAction}}()
    for (ids, bel) in b
        safe_acts[(ids[2], ids[1])] = safe_actions(pomdp, mask, bel, ids[2], ids[1]) 
    end
    probs, probs_dict = compute_probas(pomdp, mask, b)
    acts = safe_actions(mask, probs)
    # reduce(intersect, values(safe_acts))
    return acts, safe_acts
end

function compute_probas(pomdp::UrbanPOMDP, mask::SafetyMask{PedCarMDP, P}, b::Dict{I, PedCarRNNBelief}) where {P <: Policy,I}
    probs_dict = Dict{Tuple{Int64, Int64}, Vector{Float64}}()
    for (ids, bel) in b
        probs_dict[(ids[2], ids[1])] = compute_probas(pomdp, mask, bel, ids[2], ids[1])
    end
    probs = reduce((x,y) -> min.(x,y), values(probs_dict))
    return probs, probs_dict
end

function POMDPModelTools.action_info(p::DecMaskedPolicy, b::Dict)
    if isempty(b)
        return UrbanAction(2.0), (actions(pomdp), ones(n_actions(pomdp)),  Dict{Tuple{Int64, Int64}, Vector{Float64}}(), Dict{Tuple{Int64, Int64}, Vector{UrbanAction}}())
    else
        # println("Beliefs keys: ", keys(b))
        safe_acts, safe_acts_dict = safe_actions(p.problem, p.mask, b)
        # compute probas
        probs_dict = Dict{Tuple{Int64, Int64}, Vector{Float64}}()
        for (ids, bel) in b
            probs_dict[(ids[2], ids[1])] = compute_probas(p.problem, p.mask, bel, ids[2], ids[1])
        end
        probs = reduce((x,y) -> min.(x,y), values(probs_dict))
        val = actionvalues(p, b)
        act = best_action(safe_acts, val, p.problem)
        return act, (safe_acts, probs, probs_dict, safe_acts_dict)
    end
end


function POMDPs.action(policy::Union{DecMaskedPolicy, DecPolicy}, b::MultipleAgentsBelief)
    pedcar_beliefs = create_pedcar_beliefs(b.pomdp, b) #XXX is using global variable pomdp
    return action(policy, pedcar_beliefs)
end

function POMDPPolicies.actionvalues(policy::Union{DecMaskedPolicy, DecPolicy}, b::MultipleAgentsBelief)
    pedcar_beliefs = create_pedcar_beliefs(b.pomdp, b) #XXX is using global variable pomdp
    return actionvalues(policy, pedcar_beliefs)
end

# function POMDPModelTools.action_info(policy::DecMaskedPolicy, b::MultipleAgentsBelief)
#     global pomdp
#     pedcar_beliefs = create_pedcar_beliefs(pomdp, b) #XXX is using global variable pomdp
#     return action_info(policy, pedcar_beliefs)..., pedcar_beliefs, deepcopy(pomdp.models)
# end


# belief decomposition 

struct DecUpdater{P <: POMDP, I, U<:Updater} <: Updater
    problem::P
    updaters::Dict{I, U}
end

function POMDPs.update(up::DecUpdater, bold::Dict{NTuple{3, Int64}, PedCarRNNBelief}, a::UrbanAction, o::UrbanObs)
    ego, car_map, ped_map, obs_map = split_o(o, up.problem)
    augment_with_absent_state!(up.problem, car_map, ego, max(2, length(car_map)+2))
    augment_with_absent_state!(up.problem, ped_map, ego, 101 + length(ped_map))
    dec_o = create_pedcar_states(ego, car_map, ped_map, obs_map)
    # println("decomposed o: ", keys(dec_o))
    ref_up = up.updaters[(0,0,0)]
    bnew = Dict{NTuple{3, Int64}, PedCarRNNBelief}()
    for (obs_id, obs) in dec_o
        if haskey(bold, obs_id) 
            @assert haskey(up.updaters, obs_id) "KeyError: $obs_id keys in old belief: $(keys(bold)), keys in updater: $(keys(up.updaters))"# should have an associated filter 
            bnew[obs_id] = update(up.updaters[obs_id], bold[obs_id], a, obs)
        else # instantiate new filter 
            up.updaters[obs_id] = PedCarRNNUpdater(deepcopy(ref_up.models), ref_up.mdp, ref_up.pomdp) # could do something smarter than deepcopy
            reset_updater!(up.updaters[obs_id])
            init_belief = PedCarRNNBelief(Vector{Vector{Float64}}(undef, n_models), obs)
            bnew[obs_id] = update(up.updaters[obs_id], init_belief, a, obs)
        end
    end
    return bnew
end   

function augment_with_absent_state!(pomdp::UrbanPOMDP, dict::OrderedDict{Int64, Vector{Float64}}, ego::Vector{Float64}, id::Int64)
    dict[id] = get_normalized_absent_state(pomdp, ego)
    return dict 
end

function create_pedcar_states(ego, car_map, ped_map, obs_map)
    decomposed_state = Dict{NTuple{3, Int64}, Vector{Float64}}()
    for (car_id, car) in car_map
        for (ped_id, ped) in ped_map
            for (obs_id, obs) in obs_map
                decomposed_state[(car_id, ped_id, obs_id)] = vcat(ego, car, ped, obs)
            end
        end
    end
    return decomposed_state
end

struct MultipleInterpolatedBeliefsOverlay <: SceneOverlay
    beliefs::Dict{NTuple{3, Int64}, PedCarRNNBelief}
    pomdp::UrbanPOMDP
    mdp::PedCarMDP
    pedcar_pomdp::UrbanPOMDP
    models::Dict{Int64, DriverModel}
end

function AutoViz.render!(rendermodel::RenderModel, overlay::MultipleInterpolatedBeliefsOverlay , scene::Scene, env::OccludedEnv)
    for (ids, b) in overlay.beliefs 
        for j=1:length(b.predictions)
            obs = obs_to_scene(overlay.pedcar_pomdp, b.predictions[j])
            itp_overlay = InterpolationOverlay(overlay.mdp, overlay.models, obs, car_id=ids[1], ped_id=ids[2]) #,)
            render!(rendermodel, itp_overlay, scene, env)
        end
    end
end

function evaluation_loop(pomdp::UrbanPOMDP, policy::Policy, up::MultipleAgentsTracker; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    @showprogress for ep=1:n_ep
        s0 = initialstate(pomdp, rng)
        o0 = initialobs(pomdp,s0, rng)
        b0 = initialize_belief(up, o0)
        hr = HistoryRecorder(max_steps=100, rng=rng)
        hist = simulate(hr, pomdp, policy, up, b0, s0);
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = is_crash(hist.state_hist[end])#sum(hist.reward_hist .<= -1.) #+ Int(n_steps(hist) >= max_steps)
    end
    return rewards, steps, violations
end

# utility decomposition 

# struct DecomposedMask{CM <: SafetyMask, PM <: SafetyMask}
#     pomdp::UrbanPOMDP
#     car_mask::CM
#     ped_mask::PM
# end

# function safe_actions(pomdp::UrbanPOMDP, mask::DecomposedMask, o::UrbanObs)
#     s = obs_to_scene(pomdp, o)
#     action_sets = Vector{Vector{UrbanAction}}()
#     np = 0
#     nc = 0
#     current_ids = keys(pomdp.models)
#     for veh in s 
#         if veh.id == EGO_ID
#             continue
#         elseif veh.def.class == AgentClass.PEDESTRIAN
#             safe_acts = safe_actions(mask.ped_mask, s, veh.id)
#             push!(action_sets, safe_acts)
#             np += 1
#             # println("For veh $(veh.id) at $(veh.state.posF), safe actions $safe_acts")
#         elseif veh.def.class == AgentClass.CAR
#             # trick
#             veh_ = veh.id
#             if !(haskey(pomdp.models, veh.id))
#                 veh_ = veh.id == 2 ? 3 : 2
#             end
#             safe_acts = safe_actions(pomdp, mask.car_mask, s, veh_)
#             nc += 1
#             push!(action_sets, safe_acts)
#             # println("For veh $(veh.id) at $(veh.state.posF), $(veh.state.v), safe actions $safe_acts")
#         end
#     end
#     # add absent pedestrian and absent car
#     # if np < pomdp.max_peds
#     #     push!(action_sets, safe_actions(mask.ped_mask, s, 102+pomdp.max_peds))
#     # end
#     # if nc < pomdp.max_cars
#     #     push!(action_sets, safe_actions(pomdp, mask.car_mask, s, 3 + pomdp.max_cars))
#     # end
#     if isempty(action_sets)
#         return actions(pomdp)
#     end
#     # take intersection
#     acts = intersect(action_sets...)
#     if isempty(acts)
#         return UrbanAction[UrbanAction(-4.0)]
#     end
#     return acts 
# end


### Template for decomposition method 

# struct DecPolicy{P <: Policy, M <: Union{MDP, POMDP}, A} <: Policy
#     policy::P # the single agent policy
#     problem::M # the pomdp definition
#     action_map::Vector{A}
#     op # the reduction operator for utiliy fusion (e.g. sum or min)
# end


# function action_values(policy::DecPolicy, dec_belief::Dict)  # no hidden state!
#     return reduce(policy.op, action_values(policy.policy, b) for (_,b) in dec_belief)
# end

# function POMDPs.action(p::DecPolicy, b::Dict)
#     vals = action_values(p, b)
#     ai = indmax(vals)
#     return p.action_map[ai]
# end

# function action_values(p::AlphaVectorPolicy, b::SparseCat)
#     num_vectors = length(p.alphas)
#     utilities = zeros(n_actions(p.pomdp), num_vectors)
#     action_counts = zeros(n_actions(p.pomdp))
#     for i = 1:num_vectors
#         ai = actionindex(p.pomdp, p.action_map[i])
#         action_counts[ai] += 1
#         utilities[ai, i] += sparse_cat_dot(p.pomdp, p.alphas[i], b)
#     end
#     utilities ./= action_counts
#     return maximum(utilities, dims=2)
# end

# # perform dot product between an alpha vector and a sparse cat object
# function sparse_cat_dot(problem::POMDP, alpha::Vector{Float64}, b::SparseCat)
#     val = 0.
#     for (s, p) in weighted_iterator(b)
#         si = stateindex(problem, s)
#         val += alpha[si]*p
#     end
#     return val 
# end
