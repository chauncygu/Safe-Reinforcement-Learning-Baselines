rng = MersenneTwister(1)
using AutomotivePOMDPs
using MDPModelChecking
using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using DiscreteValueIteration
using ProgressMeter, Parameters, JLD

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);
             
mdp = CarMDP(env = env, vel_res=2.0, pos_res=3.0);

function MDPModelChecking.labels(mdp::CarMDP, s::CarMDPState)
    if s.crash
        return ["crash"]
    elseif s.ego.posF.s >= get_end(mdp.env.roadway[mdp.ego_goal]) &&
            get_lane(mdp.env.roadway, s.ego).tag == mdp.ego_goal
        return ["goal"]
    else
        return ["!crash", "!goal"]
    end
end

property = "!crash U goal" 

solver = ModelCheckingSolver(property=property, solver=ValueIterationSolver())

policy = solve(solver, mdp, verbose=true)

JLD.save("carmdp.jld", "policy", policy)
JLD.save("car_acc_states.jld", "accepting_states", policy.mdp.accepting_states)
