N_PROCS=56
addprocs(N_PROCS)
rng = MersenneTwister(1)
@everywhere begin 
    using POMDPs, POMDPToolbox, DiscreteValueIteration
    using AutomotivePOMDPs, AutomotiveDrivingModels
    using JLD, StaticArrays

    function DiscreteValueIteration.ind2state(mdp::PedCarMDP, si::Int64)
        n_ego = n_ego_states(mdp.env, mdp.pos_res, mdp.vel_res)
        n_ped = n_ped_states(mdp.env, mdp.pos_res, mdp.vel_ped_res)
        routes = AutomotivePOMDPs.get_car_routes(mdp.env)
        n_routes = length(routes)
        car, ped, ego = nothing, nothing, nothing
        # find route first
        ns = 0 
        route_ind = 0
        route_shift = 0
        for (i, route) in enumerate(routes)
            n_cars = n_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res)
            route_shift = ns
            ns += n_cars*n_ego*(n_ped + 1)
            if ns >= si 
                route_ind = i
                break
            end
        end
        # find car, ped, ego
        if route_ind == 0 # route was not found, car is off the grid
            si_ = si - ns # shift by all the states that were added before
            car = get_off_the_grid(mdp)
            # retrieve ped and ego
            ped_i, ego_i = ind2sub((n_ped + 1, n_ego), si_)
            ego = ind2ego(mdp.env, ego_i, mdp.pos_res, mdp.vel_res)
            if ped_i > n_ped
                ped = get_off_the_grid(mdp)
            else
                ped = ind2ped(mdp.env, ped_i, mdp.pos_res, mdp.vel_ped_res)
            end
            collision = collision =  is_colliding(Vehicle(ego, mdp.ego_type, EGO_ID), Vehicle(car, mdp.car_type, CAR_ID)) || is_colliding(Vehicle(ego, mdp.ego_type, EGO_ID), Vehicle(ped, mdp.ped_type, PED_ID)) 
            return PedCarMDPState(collision, ego, ped, car, SVector{2, LaneTag}(LaneTag(0,0), LaneTag(0, 0)))
        else
            si_ = si - route_shift
            route = routes[route_ind]
            sroute = SVector{2, LaneTag}(route[1], route[end])
            n_cars = n_car_states(mdp.env, route, mdp.pos_res, mdp.vel_res)
            # use ind2sub magic
            car_i, ped_i, ego_i = ind2sub((n_cars, n_ped + 1, n_ego), si_)
            car = ind2car(mdp.env, car_i, route, mdp.pos_res, mdp.vel_res)
            ego = ind2ego(mdp.env, ego_i, mdp.pos_res, mdp.vel_res)
            if ped_i > n_ped
                ped = get_off_the_grid(mdp)
            else
                ped = ind2ped(mdp.env, ped_i, mdp.pos_res, mdp.vel_ped_res)
            end
            collision =  is_colliding(Vehicle(ego, mdp.ego_type, EGO_ID), Vehicle(car, mdp.car_type, CAR_ID)) || is_colliding(Vehicle(ego, mdp.ego_type, EGO_ID), Vehicle(ped, mdp.ped_type, PED_ID)) 
            return PedCarMDPState(collision, ego, ped, car, sroute)
        end
    end

end # @everywhere



params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env=env, pos_res=2.0, vel_res=2.0, ped_birth=0.7, car_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))

# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

solver = ParallelSynchronousValueIterationSolver(n_procs=N_PROCS, max_iterations=4, belres=1e-4, include_Q=true, verbose=true)
policy_file = "pc_util_sync.jld"
#policy = ValueIterationPolicy(mdp, include_Q=true)
if isfile(policy_file)
  data = load(policy_file)
  solver.init_util = data["util"]
end
policy = solve(solver, mdp)
JLD.save(policy_file, "util", policy.util, "qmat", policy.qmat, "pol", policy.policy)

solver.init_util = policy.util
policy = solve(solver, mdp) # resume

JLD.save(policy_file, "util", policy.util, "qmat", policy.qmat, "pol", policy.policy)
JLD.save("pedcar_policy3.jld", "policy", policy)
