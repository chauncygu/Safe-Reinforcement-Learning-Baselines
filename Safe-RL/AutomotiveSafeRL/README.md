# Safe Planning Under Uncertainty for Autonomous Driving

Experiments on safe planning under uncertainty for autonomous driving. 
This code base combines reachability analysis, reinforcement learning, and decomposition methods to compute safe and efficient policies for autonomous vehicles.

**Reference**: M. Bouton, A. Nakhaei, K. Fujimura, and M. J. Kochenderfer, “Safe reinforcement learning with scene decomposition for navigating complex urban environments,” in IEEE Intelligent Vehicles Symposium (IV), 2019. 

## Installation

For julia version >1.1, it is recommended to add the SISL registry and JuliaPOMDP registry first:
```julia
pkg> registry add https://github.com/sisl/Registry
pkg> registry add https://github.com/JuliaPOMDP/Registry
pkg> add https://github.com/MaximeBouton/PedCar.jl
pkg> add https://github.com/MaximeBouton/AutomotiveSafeRL
```

To install all the dependencies manually, run the following in the Julia REPL:

```julia 
using Pkg
Pkg.add(PackageSpec(url="https://github.com/sisl/Vec.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/Records.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveDrivingModels.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutoViz.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutoUrban.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotiveSensors.jl"))
Pkg.add(PackageSpec(url="https://github.com/JuliaPOMDP/RLInterface.jl"))
Pkg.add(PackageSpec(url="https://github.com/JuliaPOMDP/DeepQLearning.jl"))
Pkg.add(PackageSpec(url="https://github.com/sisl/AutomotivePOMDPs.jl"))
Pkg.add(PackageSpec(url="https://github.com/MaximeBouton/POMDPModelChecking.jl"))
``` 

## Folder structure

- `src/` contains the implementation of the safe RL policy and the decomposition method.
- `RNNFiltering/` contains data_generation and training_script for the ensemble RNN belief updater
- `training_scripts/` contains training scripts for the safe RL and RL policies
- `evaluation/` contains evaluation scripts to evaluate RL, safe RL, and baseline policies.
- `notebooks/` contains jupyter notebook for visualization and debugging. 


## Code to run

To visualize any of the policy use `notebooks/interactive_evaluation.ipynb`

For a detailed description of the evaluation scenarios run `notebooks/evaluation_scenarios.ipynb`

Other notebooks are used for prototyping and debugging.

All scripts used to run the experiments in the paper are available, most of them have command line arguments. Check those arguments to see what can be changed.

**Solving for the safety mask using Model Checking**

Run  `training_scripts/pedcar_vi.jl` to compute the safety mask using value iteration. *This part is computationally expensive and requires parallelization over many cores*.

**Training an RL agent**

Run  `training_scripts/pedcar_dqn.jl` to train an RL agent with or without the safety mask.

**Training the belief updater**

First, run the `RNNFiltering/generate_dataset.jl` to creat synthetic data to train the RNN updater on. 
Then run `RNNFiltering/bagging_training.jl` to train one RNN. Look at the bash script `RNNFiltering/train.sh` to check how to properly set the seed. 

**Evaluating the algorithm**
Run  `evaluation/evaluation.jl` or `evaluation/parallel_evaluation.jl` to evaluate the algorithm (this requires a solved VI mask, trained RL policy and trained RNN filer).


## Main Dependencies

- AutomotivePOMDPs.jl contains all the driving scenario and MDP models
- POMDPModelChecking.jl 
- DeepQLearning.jl (Flux.jl backend)
