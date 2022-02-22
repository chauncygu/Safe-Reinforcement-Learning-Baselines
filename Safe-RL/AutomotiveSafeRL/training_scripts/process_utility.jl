using AutomotiveDrivingModels
using AutomotivePOMDPs
using PedCar
using JLD2
using FileIO
using POMDPs
using DiscreteValueIteration
using ProgressMeter
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--load"
        arg_type=String
        default="pedcar_utility.jld2"
    "--save"
        arg_type=String 
        default = "pedcar_utility_processed.jld2"
end
parsed_args = parse_args(ARGS, s)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =[VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env=env, pos_res=2.0, vel_res=2., ped_birth=0.7, car_birth=0.7);
init_transition!(mdp);

# Load VI data for maksing
@time state_space = states(mdp);
vi_data = load(parsed_args["load"])
function process_utility(mdp, state_space, vi_data)
    @showprogress for s in state_space
        if !s.crash && isterminal(mdp, s)
            si = stateindex(mdp, s)
            vi_data["util"][si] = 1.0
            vi_data["qmat"][si, :] = ones(n_actions(mdp))
        end
    end
    policy = ValueIterationPolicy(mdp, vi_data["qmat"], vi_data["util"], vi_data["pol"]);
    util = policy.util
    qmat = policy.qmat
    pol = policy.policy 
    return util, qmat, pol
end
util, qmat, pol = process_utility(mdp, state_space, vi_data)
@save parsed_args["save"] util qmat pol