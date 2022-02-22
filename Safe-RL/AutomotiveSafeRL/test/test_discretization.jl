

function test_ego_space(env::UrbanEnv, pos_res::Float64, vel_res::Float64)
    ego_space = get_ego_states(env, pos_res, vel_res);
    for (i, s) in enumerate(ego_space)
        if i != ego_stateindex(env, s, pos_res, vel_res)
            return false
        end
    end
    return true
end

function test_car_space(env::UrbanEnv, pos_res::Float64, vel_res::Float64)
    car_space = get_car_states(env, pos_res, vel_res);
    for (i, s) in enumerate(car_space)
        if i != car_stateindex(env, s, pos_res, vel_res)
            return false
        end
    end
    return true
end

function test_car_indexing(env::UrbanEnv, pos_res::Float64, vel_res::Float64)
    routes = get_car_routes(env)
    for route in routes
        car_states = get_car_states(env, route, pos_res, vel_res)
        for (i, car) in enumerate(car_states)
            car_i = car_stateindex(env, car, route, pos_res, vel_res)
            if car_i != i
                return false
            end
        end
        @assert n_car_states(env, route, pos_res, vel_res) == length(car_states)
    end
    return true
end

function test_ped_space(env::UrbanEnv, pos_res::Float64, vel_res::Float64)
    ped_space = get_ped_states(env, pos_res, vel_res);
    for (i, s) in enumerate(ped_space)
        if i != ped_stateindex(env, s, pos_res, vel_res)
            return false
        end
    end
    return true
end



params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [10.0, 10., 10.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

pos_res = 3.0
vel_res = 2.0
vel_ped_res = 1.0

@test test_ego_space(env, pos_res, vel_res)
@test test_car_space(env, pos_res, vel_res)
@test test_car_indexing(env, pos_res, vel_ped_res)
@test test_ped_space(env, pos_res, vel_ped_res)
