@everywhere begin
    using POMDPs, POMDPToolbox, DiscreteValueIteration
    using AutomotivePOMDPs, AutomotiveDrivingModels
end
rng = MersenneTwister(1)

params = UrbanParams(nlanes_main=1,
                     crosswalk_pos =  [VecSE2(6, 0., pi/2), VecSE2(-6, 0., pi/2), VecSE2(0., -5., 0.)],
                     crosswalk_length =  [14.0, 14., 14.0],
                     crosswalk_width = [4.0, 4.0, 3.1],
                     stop_line = 22.0)
env = UrbanEnv(params=params);

mdp = PedCarMDP(env=env, pos_res=6.0, vel_res=3.0, ped_birth=0.7, ped_type=VehicleDef(AgentClass.PEDESTRIAN, 1.0, 3.0))
# reachability analysis
mdp.collision_cost = 0.
mdp.γ = 1.
mdp.goal_reward = 1.

solver = ParallelValueIterationSolver(n_procs=8, max_iterations=4, belres=1e-4)
policy = solve(solver, mdp, verbose=true)
