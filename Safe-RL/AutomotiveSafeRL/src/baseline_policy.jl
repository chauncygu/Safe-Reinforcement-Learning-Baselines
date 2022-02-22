struct EgoBaseline <: Policy
    pomdp::UrbanPOMDP
    model::DriverModel
end

mutable struct MaskedEgoBaseline <: Policy
    pomdp::UrbanPOMDP
    pedcar_pomdp::UrbanPOMDP
    model::DriverModel
    mask::SafetyMask
    sa::Vector{UrbanAction}
end

function POMDPs.action(policy::EgoBaseline, o::UrbanObs)
    s = obs_to_scene(policy.pomdp, o)
    return action(policy, s)
end

function POMDPs.action(policy::EgoBaseline, s::Scene)
    observe!(policy.model, s, policy.pomdp.env.roadway, EGO_ID)
    acts = [a.acc for a in ordered_actions(policy.pomdp)]
    ai = argmin(abs.(policy.model.a.a_lon .- acts))
    return ordered_actions(policy.pomdp)[ai]
end

function POMDPModelTools.action_info(policy::EgoBaseline, o::UrbanObs)
    a = action(policy, o)
    return a, (policy.model.a.a_lon, deepcopy(policy.model))
end

function POMDPs.action(policy::EgoBaseline, b::MultipleAgentsBelief)
    s = most_likely_scene(policy.pomdp, b)
    action(policy, s)
end

function POMDPModelTools.action_info(policy::EgoBaseline, b::MultipleAgentsBelief)
    s = most_likely_scene(policy.pomdp, b)
    return action(policy, s), (s, policy.model.a.a_lon, deepcopy(policy.model))
end

function POMDPs.action(policy::MaskedEgoBaseline, s::Scene)
    observe!(policy.model, s, policy.pomdp.env.roadway, EGO_ID)
    acts = [a.acc for a in ordered_actions(pomdp)]
    ai = argmin(abs.(policy.model.a.a_lon .- acts))
    a = ordered_actions(pomdp)[ai]
    if a ∈ policy.sa
        return a 
    else 
        ai = argmax([a.acc for a in policy.sa])
        return policy.sa[ai]
    end
end

function POMDPs.action(policy::MaskedEgoBaseline, o::UrbanObs)
    policy.sa = safe_actions(policy.pomdp, policy.mask, o)
    s = obs_to_scene(policy.pomdp, o)
    return action(policy, s)
end

function POMDPs.action(policy::MaskedEgoBaseline, b::MultipleAgentsBelief)
    s = most_likely_scene(policy.pomdp, b)
    observe!(policy.model, s, policy.pomdp.env.roadway, EGO_ID)
    acts = [a.acc for a in ordered_actions(policy.pomdp)]
    ai = argmin(abs.(policy.model.a.a_lon .- acts))
    a = ordered_actions(policy.pomdp)[ai]
    pedcar_beliefs = create_pedcar_beliefs(b.pomdp, b)
    if isempty(pedcar_beliefs)
        return UrbanAction(2.0)
    end
    policy.sa, _ = safe_actions(policy.pedcar_pomdp, policy.mask, pedcar_beliefs)
    if a ∈ policy.sa
        return a 
    else 
        ai = argmax([a.acc for a in policy.sa])
        return policy.sa[ai]
    end
end

function POMDPModelTools.action_info(policy::MaskedEgoBaseline, o::UrbanObs)
    a = action(policy, o)
    return a, (policy.model.a.a_lon, policy.sa)
end

function get_ego_baseline_model(env::UrbanEnv)
    route = [env.roadway[tag] for tag in [LaneTag(6,1), LaneTag(13,1), LaneTag(2,1)]]
    intersection_entrances = get_start_lanes(env.roadway)
    push!(intersection_entrances, env.roadway[LaneTag(6,1)])
    intersection_exits = get_exit_lanes(env.roadway)
    intersection=Lane[route[1], route[2]]
    navigator = RouteFollowingIDM(route=route, a_max=2., σ=0., v_des=8.0)
    intersection_driver = TTCIntersectionDriver(navigator = navigator,
                                            intersection = intersection,
                                            intersection_pos = VecSE2(env.params.inter_x,
                                                                      env.params.inter_y),
                                            stop_delta = maximum(env.params.crosswalk_width),
                                            accel_tol = 0.,
                                            priorities = env.priorities,
                                            ttc_threshold = (env.params.x_max - env.params.inter_x)/env.params.speed_limit
                                            )
    crosswalk_drivers = Vector{CrosswalkDriver}(undef, length(env.crosswalks))
    for i=1:length(env.crosswalks)
        cw_conflict_lanes = get_conflict_lanes(env.crosswalks[i], env.roadway)
        crosswalk_drivers[i] = CrosswalkDriver(navigator = navigator,
                                crosswalk = env.crosswalks[i],
                                conflict_lanes = cw_conflict_lanes,
                                intersection_entrances = intersection_entrances,
                                yield=!isempty(intersect(cw_conflict_lanes, route))
                                )
        # println(" yield to cw ", i, " ", crosswalk_drivers[i].yield)
    end
    return UrbanDriver(navigator=navigator,
                    intersection_driver=intersection_driver,
                    crosswalk_drivers=crosswalk_drivers
                    )
end
