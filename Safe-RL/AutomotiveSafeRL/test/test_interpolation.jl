#TODO make this dependent on env, pos_res, vel_res instead of MDP
function test_ego_interpolation(mdp::PedMDP)
    ego_state_space = get_ego_states(mdp.env, mdp.pos_res, mdp.vel_res)
    v_space = get_car_vspace(mdp.env, mdp.vel_res)
    for s in ego_state_space
        states, weights = interpolate_state(s, v_space)
        if !(length(states) == 1 &&
             weights[1] == 1.0 &&
             states[1] ∈ ego_state_space)
            return false
        end
    end
    return true
end

function test_car_interpolation(mdp::PedMDP)
    car_state_space = get_car_states(mdp.env, mdp.pos_res, mdp.vel_res)
    v_space = get_car_vspace(mdp.env, mdp.vel_res)
    for s in car_state_space
        states, weights = interpolate_state(s, v_space)
        if !(length(states) == 1 &&
             weights[1] == 1.0 &&
             states[1] ∈ car_state_space)
            return false
        end
    end
    return true
end


function test_ped_interpolation(mdp::PedMDP)
    ped_state_space = get_ped_states(mdp.env, mdp.pos_res, mdp.vel_ped_res)
    v_space = get_ped_vspace(mdp.env, mdp.vel_ped_res)
    for s in ped_state_space
        states, weights = interpolate_pedestrian(s, v_space)
        if !(length(states) == 1 &&
             weights[1] == 1.0 &&
             states[1] ∈ ped_state_space)
            return false
        end
    end
    return true
end

function random_ego_interpolation(rng::AbstractRNG)
    lanes = get_ego_route(mdp.env)
    lane = lanes[rand(rng, 1:length(lanes))]
    s = rand(rng, 0:0.1:get_end(mdp.env.roadway[lane]))
    v = rand(rng, 0:0.1:mdp.env.params.speed_limit)
    ego_state = VehicleState(Frenet(mdp.env.roadway[lane], s), mdp.env.roadway, v)
    states, probs = interpolate_state(ego_state, get_car_vspace(mdp.env, mdp.vel_res))
    scene = Scene()
    for (i, itp_state) in enumerate(states)
        push!(scene, Vehicle(itp_state, mdp.ego_type, i+1))
    end
    push!(scene, Vehicle(ego_state, mdp.ego_type, 1))
    render(scene, env, cam=FitToContentCamera(0.), car_colors=get_colors(scene))
end

function random_ped_interpolation(rng::AbstractRNG)
    lanes = get_ped_lanes(mdp.env)
    lane = lanes[rand(rng, 1:length(lanes))]
    s = rand(rng, 0:0.1:get_end(mdp.env.roadway[lane]))
    v = rand(rng, 0:0.1:2.0)
    ped_state = VehicleState(Frenet(mdp.env.roadway[lane], s), mdp.env.roadway, v)
    states, probs = interpolate_state(ped_state, get_car_vspace(mdp.env, mdp.vel_res))
    scene = Scene()
    for (i, itp_state) in enumerate(states)
        push!(scene, Vehicle(itp_state, mdp.ped_type, i+1))
    end
    push!(scene, Vehicle(ped_state, mdp.ped_type, 1))
    render(scene, env, cam=FitToContentCamera(0.), car_colors=get_colors(scene))
end


function random_car_interpolation(rng::AbstractRNG)
    lanes = get_car_lanes(mdp.env)
    lane = lanes[rand(rng, 1:length(lanes))]
    s = rand(rng, 0:0.1:get_end(mdp.env.roadway[lane]))
    v = rand(rng, 0:0.1:2.0)
    car_state = VehicleState(Frenet(mdp.env.roadway[lane], s), mdp.env.roadway, v)
    states, probs = interpolate_state(car_state, get_car_vspace(mdp.env, mdp.vel_res))
    scene = Scene()
    for (i, itp_state) in enumerate(states)
        push!(scene, Vehicle(itp_state, VehicleDef(), i+1))
    end
    push!(scene, Vehicle(car_state, VehicleDef(), 1))
    render(scene, env, cam=FitToContentCamera(0.), car_colors=get_colors(scene))
end
