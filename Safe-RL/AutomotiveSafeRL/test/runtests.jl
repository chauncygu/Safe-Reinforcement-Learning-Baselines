using Base.Test
using Parameters
#dep
include("../AutomotivePOMDPs/AutomotivePOMDPs.jl")
using AutomotivePOMDPs
using POMDPs, POMDPToolbox, AutoViz, AutomotiveDrivingModels, Reel
using GridInterpolations, StaticArrays
include("mdp_models/discretization.jl")
include("mdp_models/pedestrian_mdp/pomdp_types.jl")
include("mdp_models/pedestrian_mdp/state_space.jl")

rng = MersenneTwister(1)

include("test_discretization.jl")
include("test_pedestrian_mdp.jl")
include("test_interpolation.jl")
