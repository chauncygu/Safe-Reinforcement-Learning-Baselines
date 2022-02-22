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
             
mdp = PedMDP(env = env, vel_res=2.0, pos_res=3.0);

function MDPModelChecking.labels(mdp::PedMDP, s::PedMDPState)
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

ltl2tgba(solver.property, solver.automata_file)
autom_type = automata_type(solver.automata_file)
automata = nothing
if autom_type == "Buchi"
    automata = hoa2buchi(solver.automata_file)
elseif autom_type == "Rabin"
    automata = hoa2rabin(solver.automata_file)
end
pmdp = nothing
if isa(mdp, POMDP)
    pmdp = ProductPOMDP(mdp, automata)
else
    pmdp = ProductMDP(mdp, automata) # build product mdp x automata
end

function goal_accepting!(mdp::ProductMDP{PedMDPState, UrbanAction, Q, T}; verbose::Bool=false) where {Q,T}
    for s in states(mdp)
        if isterminal(mdp.mdp, s.s) && !s.s.crash && s.q == 1
            push!(mdp.accepting_states, s)
        end
    end
    return accepting_states
end
    
goal_accepting!(pmdp)

if isempty(pmdp.accepting_states)
    accepting_states!(pmdp, verbose=verbose) # compute the maximal end components via a graph analysis
end
policy = solve(solver.solver, pmdp, verbose=true) # solve using your favorite method
JLD.save("ped_product_vi.jld", "policy", policy)

