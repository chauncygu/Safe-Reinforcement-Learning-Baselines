rng = MersenneTwister(1)
@everywhere begin
    using AutomotivePOMDPs
    using MDPModelChecking
    using GridInterpolations, StaticArrays, POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
    using DiscreteValueIteration
    using ProgressMeter, Parameters, JLD
end
params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = CarMDP(env = env, pos_res=2., vel_res=3., car_birth=0.7)

# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

solver = ParallelValueIterationSolver(n_procs=7)

policy = solve(solver, mdp, verbose=true)
JLD.save("car_until.jld", "util", policy.util, "qmat", policy.qmat, "policy", policy.policy)
