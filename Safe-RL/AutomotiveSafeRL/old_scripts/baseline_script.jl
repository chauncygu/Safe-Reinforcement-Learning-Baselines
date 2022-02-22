rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DeepQLearning, DeepRL
using DiscreteValueIteration
using ProgressMeter, Parameters, JLD

include("util.jl")

# define baseline policy 

function evaluation_loop(pomdp::UrbanPOMDP, policy::UrbanDriverPolicy; n_ep::Int64 = 1000, max_steps::Int64 = 500, rng::AbstractRNG = Base.GLOBAL_RNG)
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    up = FastPreviousObservationUpdater{obs_type(pomdp)}()
    for ep=1:n_ep
        reset_policy!(policy)
        s0 = initialstate(pomdp, rng)
        o0 = generate_o(pomdp, s0, rng)
        b0 = initialize_belief(up, o0)
        hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        hist = simulate(hr, pomdp, policy, up, b0, s0);
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        # violations[ep] = sum(hist.reward_hist .< 0.)
        violations[ep] = is_crash(hist.state_hist[end])#sum(hist.reward_hist .<= -1.) #+ Int(n_steps(hist) >= max_steps)
    end
    return rewards, steps, violations
end

mutable struct UrbanDriverPolicy <: Policy
    pomdp::UrbanPOMDP
    model::UrbanDriver
end

function UrbanDriverPolicy(pomdp::UrbanPOMDP)
    route = [env.roadway[l] for l in get_ego_route(pomdp.env)]
    intersection_entrances = get_start_lanes(pomdp.env.roadway)
    if !(route[1] ∈ intersection_entrances)
        intersection = Lane[]
        intersection_exits = Lane[]
    else
        intersection_exits = get_exit_lanes(pomdp.env.roadway)
        intersection=Lane[route[1], route[2]]
    end
    navigator = RouteFollowingIDM(route=route, a_max=2.)
    intersection_driver = StopIntersectionDriver(navigator= navigator,
                                                intersection=intersection,
                                                intersection_entrances = intersection_entrances,
                                                intersection_exits = intersection_exits,
                                                stop_delta=maximum(pomdp.env.params.crosswalk_width),
                                                accel_tol=0.,
                                                priorities = pomdp.env.priorities)
    crosswalk_drivers = Vector{CrosswalkDriver}(length(pomdp.env.crosswalks))
    # println("adding veh ", new_car.id)
    for i=1:length(pomdp.env.crosswalks)
        cw_conflict_lanes = get_conflict_lanes(pomdp.env.crosswalks[i], pomdp.env.roadway)
        crosswalk_drivers[i] = CrosswalkDriver(navigator = navigator,
                                crosswalk = pomdp.env.crosswalks[i],
                                conflict_lanes = cw_conflict_lanes,
                                intersection_entrances = intersection_entrances,
                                yield=!isempty(intersect(cw_conflict_lanes, route))
                                )
        # println(" yield to cw ", i, " ", crosswalk_drivers[i].yield)
    end
    model = UrbanDriver(navigator=navigator,
                        intersection_driver=intersection_driver,
                        crosswalk_drivers=crosswalk_drivers
                        )
    return UrbanDriverPolicy(pomdp, model)
end

function POMDPs.action(policy::UrbanDriverPolicy, s::Scene)
    observe!(policy.model, s, policy.mdp.env.roadway, EGO_ID)
    return policy.model.a
end

function reset_policy!(policy::UrbanDriverPolicy)
    reset_hidden_state!(policy.model)
end

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params)

pomdp = UrbanPOMDP(env=env,
                   ego_goal = LaneTag(2, 1),
                   max_cars=1, 
                   max_peds=0, 
                   car_birth=0.3, 
                   ped_birth=0.7, 
                   obstacles=false, # no fixed obstacles
                   lidar=false,
                   pos_obs_noise = 0., # fully observable
                   vel_obs_noise = 0.);

# evaluate resulting policy
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, policy, n_ep=1000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)