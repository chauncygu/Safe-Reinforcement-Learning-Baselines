# Synthesize and Test Cruise Control Shields

Synthesize shields and test their safety on a random agent.
By shields I mean a nondeterministic strategy that can be used to shield a learning agent or another strategy.

As opposed to the similar Bouncing Ball experiment, only one random agent will be used. 
It is the random agent with uniform chance of picking any action. 

Shields are synthesised using the "barbaric" reachability method only.
The barbaric method makes use of a sampling-based method to under-approximate the possible outcomes of the system. This is a quick-and-dirty solution to the reachability problem, and will be tested here is whether it works in practice. 

Everything is tied together in the file `Run Experiment.jl`. Run as `julia "tab-CCSynthesis/Run Experiment.jl"` from within the ReproducibilityPackage folder. 

It makes use of files `CC Synthesize Set of Shields.jl` and `CC Statistical Checking of Shield.jl` which in turn depend on code  found in `Shared Code`. 

The files are Pluto Notebooks, which by their nature are also valid standalone julia scripts. 
