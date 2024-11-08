# Synthesize and Test Shields

Synthesize shields and test their safety against different random agent. 
"Shield" is used as shorthand for a nondeterministic strategy that can be used to shield a learning agent or another strategy.

A random agent is defined by it's `hit_chance` such that it will choose randomly between actions `(hit, nohit)` with probabilities `(1-hit_chance, hit_chance)`. 

Shields are synthesised using either a "barbaric" or "rigorous" reachability method. 
The rigorous method makes use of the library `ReachabilityAnalysis.jl` to over-approximate possible outcomes of the system. This gives theoretical guarantees for the safety, at the cost of more compute time and a less optimistic shield. 
The barbaric method makes use of a sampling-based method to under-approximate the possible outcomes of the system. This is a quick-and-dirty solution to the reachability problem, and will be tested here is whether it works in practice. 

Everything is tied together in the file `Run Experiment.jl`. Run as `julia "Run Experiment.jl"` from within this folder. 

Some of files are Pluto Notebooks, which by their nature are also valid standalone julia scripts. 
