#N_PROCS = 40
#addprocs(N_PROCS)

#@everywhere begin
    using POMDPs, POMDPToolbox, DiscreteValueIteration, MDPModelChecking
    using AutomotiveDrivingModels, AutomotivePOMDPs
    using LocalApproximationValueIteration
    using PedCar
    using Reel, AutoViz
    using ProgressMeter
    using JLD


    include("masking.jl")
    include("util.jl")
    include("render_helpers.jl")

    params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =[VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
    env = UrbanEnv(params=params);

    mdp = PedCarMDP(env=env, ped_birth=0.7, car_birth=0.7, pos_res=2.0, vel_res=2.);
    init_transition!(mdp)
## Load VI data for maksing
#state_space = states(mdp);
vi_data = JLD.load("pc_util_inter.jld")
@showprogress for s in state_space
     if !s.crash && isterminal(mdp, s)
         si = stateindex(mdp, s)
         vi_data["util"][si] = 1.0
         vi_data["qmat"][si, :] = ones(n_actions(mdp))
     end
end
policy = ValueIterationPolicy(mdp, vi_data["qmat"], vi_data["util"], vi_data["pol"]);
    JLD.save("pc_util_processed.jld", "util", policy.util, "qmat", policy.qmat, "pol", policy.policy)

    #vi_data = JLD.load("pc_util_processed.jld")
    #policy = ValueIterationPolicy(mdp, vi_data["qmat"], vi_data["util"], vi_data["pol"]);

    threshold = 0.9999
    mask = SafetyMask(mdp, policy, threshold);
#end #@everywhere
rng = MersenneTwister(1)
rand_pol = MaskedEpsGreedyPolicy(mdp, 1.0, mask, rng);
println("Evaluation in discretized environment: \n ")
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(mdp, rand_pol, n_ep=10000, max_steps=100, rng=rng);
print_summary(rewards_mask, steps_mask, violations_mask)

pomdp = UrbanPOMDP(env=env,
                    ego_goal = LaneTag(2, 1),
                    max_cars=1, 
                    max_peds=1, 
                    car_birth=0.7, 
                    ped_birth=0.7, 
                    obstacles=false, # no fixed obstacles
                    lidar=false,
                    pos_obs_noise = 0., # fully observable
                    vel_obs_noise = 0.,
                    ego_start=20);
rand_pol = RandomMaskedPOMDPPolicy(mask, pomdp, MersenneTwister(1));


println("\n Evaluation in continuous environment: \n")
flush(STDOUT)
@time rewards_mask, steps_mask, violations_mask = evaluation_loop(pomdp, rand_pol, n_ep=10000, max_steps=100, rng=MersenneTwister(1));
print_summary(rewards_mask, steps_mask, violations_mask)
flush(STDOUT)
