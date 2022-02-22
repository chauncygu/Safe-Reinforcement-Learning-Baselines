function evaluation_loop(pomdp::UrbanPOMDP, policy::Policy, up::Updater, static_spec!::Function, initialscene::Function; n_ep::Int64=100, max_steps::Int64=400, rng::AbstractRNG = MersenneTwister(1))
    rewards = zeros(n_ep)
    steps = zeros(n_ep)
    violations = zeros(n_ep)
    static_spec!(pomdp)
    @showprogress for ep=1:n_ep
        s0 = initialscene(pomdp, rng)
        o0 = generate_o(pomdp, s0, rng)
        b0 = initialize_belief(up, o0)
        hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        hist = simulate(hr, pomdp, policy, up, b0, s0);
        rewards[ep] = discounted_reward(hist)
        steps[ep] = n_steps(hist)
        violations[ep] = is_crash(hist.state_hist[end])
    end
    return rewards, steps, violations
end

const TURN_RIGHT = SVector(LaneTag(3,1), LaneTag(5,1))
const STRAIGHT_FROM_RIGHT = SVector(LaneTag(3,1), LaneTag(4,1))
const STRAIGHT_FROM_LEFT = SVector(LaneTag(1,1), LaneTag(2,1))
const TURN_LEFT = SVector(LaneTag(1,1), LaneTag(5,1))
const RIGHT_OBSTACLE = ConvexPolygon([VecE2(-26.875, -7.500),VecE2(-8.125, -7.500),VecE2(-8.125, -3.000),VecE2(-26.875, -3.000)], 4)
const LEFT_OBSTACLE =  ConvexPolygon([VecE2(8.125, -7.500), VecE2(26.875, -7.500), VecE2(26.875, -3.000), VecE2(8.125, -3.000)], 4)
const MAX_SPEED = 8.0
const ROUTES = [TURN_RIGHT, TURN_LEFT, STRAIGHT_FROM_LEFT, STRAIGHT_FROM_RIGHT]
const MAX_PEDESTRIAN_SPEED = 2.0

#=
Scenario 1.1 noise robustness
only one vehicle
=#

function set_static_spec_scenario_1_1!(pomdp::UrbanPOMDP)
    empty_obstacles!(pomdp.env)
    pomdp.max_cars = 1
    pomdp.ped_birth = 0.
    pomdp.car_birth = 0.
end

function initialscene_scenario_1_1(pomdp::UrbanPOMDP, rng::AbstractRNG)
    route = rand(rng, [STRAIGHT_FROM_RIGHT, STRAIGHT_FROM_LEFT])
    roadway = pomdp.env.roadway
    car_s0 = rand(rng,0.:get_end(roadway[route[1]]))
    car_v0 = rand(rng, 0.:MAX_SPEED)
    car_posF = Frenet(roadway[route[1]], car_s0)
    car = Vehicle(VehicleState(car_posF, roadway, car_v0), pomdp.car_type, 2)
    s0 = Scene()
    push!(s0, car)
    pomdp.models[2] = pomdp.car_models[route]
    push!(s0, initial_ego(pomdp, rng))
    return s0
end

#=
Scenario 1.2 occlusion robustness
only one vehicle 
=#

function set_static_spec_scenario_1_2!(pomdp::UrbanPOMDP)
    # set obstacles
    pomdp.max_obstacles = 1
end

function initialscene_scenario_1_2(pomdp::UrbanPOMDP, rng::AbstractRNG)
    right = rand() > 0.5
    if right 
        pomdp.env.obstacles = [RIGHT_OBSTACLE]
        route = STRAIGHT_FROM_RIGHT
    else
        pomdp.env.obstacles = [LEFT_OBSTACLE]
        route = STRAIGHT_FROM_LEFT
    end
    roadway = pomdp.env.roadway
    car_s0 = rand(rng,0.:get_end(roadway[route[1]]))
    car_v0 = rand(rng, 0.:MAX_SPEED)
    car_posF = Frenet(roadway[route[1]], car_s0)
    car = Vehicle(VehicleState(car_posF, roadway, car_v0), pomdp.car_type, CAR_ID)
    s0 = Scene()
    push!(s0, car)
    pomdp.models[2] = pomdp.car_models[route]
    push!(s0, initial_ego(pomdp, rng))
    return s0
end

#=
Scenario 2.1 interaction 
car / pedestrian interaction 
=# 

function set_static_spec_scenario_2_1!(pomdp::UrbanPOMDP)
    pomdp.sensor = PerfectSensor()
    empty_obstacles!(pomdp.env)
    pomdp.max_cars = 1
    pomdp.max_peds = 1
end

function initialscene_scenario_2_1(pomdp::UrbanPOMDP, rng::AbstractRNG)
    route = rand(rng, ROUTES)
    roadway = pomdp.env.roadway
    car_s0 = rand(rng,0.:get_end(roadway[route[1]]))
    car_v0 = rand(rng, 0.:MAX_SPEED)
    car_posF = Frenet(roadway[route[1]], car_s0)
    car = Vehicle(VehicleState(car_posF, roadway, car_v0), pomdp.car_type, CAR_ID)

    crosswalk = rand(rng, pomdp.env.crosswalks)
    ped_s0 = rand(rng, 0.:get_end(crosswalk))
    ped_v0 = rand(rng, 0.:MAX_PEDESTRIAN_SPEED)
    ped_posF = Frenet(crosswalk, ped_s0) # choose between 17, 18, 19
    ped = Vehicle(VehicleState(ped_posF, pomdp.env.roadway, ped_v0), pomdp.ped_type, PED_ID)

    s0 = Scene()
    push!(s0, car)
    push!(s0, ped)
    pomdp.models[CAR_ID] = pomdp.car_models[route]
    new_ped_conflict_lanes = get_conflict_lanes(crosswalk, pomdp.env.roadway)
    pomdp.models[PED_ID] = IntelligentPedestrian(dt = pomdp.ΔT, crosswalk=crosswalk, conflict_lanes=new_ped_conflict_lanes)    
    push!(s0, initial_ego(pomdp, rng))
end

#=
Scenario 2.2 interaction 
car / car interaction
=# 

function set_static_spec_scenario_2_2!(pomdp::UrbanPOMDP)
    pomdp.sensor = PerfectSensor()
    empty_obstacles!(pomdp.env)
    pomdp.max_cars = 1
end

function initialscene_scenario_2_2(pomdp::UrbanPOMDP, rng::AbstractRNG)
    route = TURN_LEFT
    roadway = pomdp.env.roadway
    car_s0 = rand(rng,0.:get_end(roadway[route[1]]))
    car_v0 = rand(rng, 0.:MAX_SPEED)
    car_posF = Frenet(roadway[route[1]], car_s0)
    car = Vehicle(VehicleState(car_posF, roadway, car_v0), pomdp.car_type, 2)

    s0 = Scene()
    push!(s0, car)
    pomdp.models[2] = pomdp.car_models[route]
    push!(s0, initial_ego(pomdp, rng))
end

#= 
Scenario 3.3 scalability
=#

function set_static_spec_scenario_3!(pomdp::UrbanPOMDP)
    pomdp.max_cars = 3.
    pomdp.max_peds = 3.
    pomdp.car_birth = 0.1
    pomdp.ped_birth = 0.1
    pomdp.max_obstacles = 1
    pomdp.obs_dist = ObstacleDistribution(pomdp.env, 
                                upper_obs_pres_prob=0., 
                                left_obs_pres_prob=1.0, 
                                right_obs_pres_prob=1.0)
end

function initialscene_scenario_3(pomdp::UrbanPOMDP, rng::AbstractRNG)
    return initialstate(pomdp, rng)
end
