
function test_stateindexing(mdp::CarMDP)
    state_space = states(mdp)
    for (i, s) in enumerate(state_space)
        if i != stateindex(mdp, s)
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

mdp = CarMDP(env = env);

@test test_stateindexing(mdp)
