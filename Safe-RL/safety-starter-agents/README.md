**Status:** Archive (code is provided as-is, no updates expected)

# Safety Starter Agents

A companion repo to the paper "Benchmarking Safe Exploration in Deep Reinforcement Learning," containing a variety of unconstrained and constrained RL algorithms.

This repo contains the implementations of PPO, TRPO, PPO-Lagrangian, TRPO-Lagrangian, and CPO used to obtain the results in the "Benchmarking Safe Exploration" paper, as well as experimental implementations of SAC and SAC-Lagrangian not used in the paper.

Note that the PPO implementations here follow the convention from [Spinning Up](https://spinningup.openai.com) rather than [Baselines](https://www.github.com/openai/baselines): they use the early stopping trick, omit observation and reward normalization, and do not use the clipped value loss, among other potential diffs. As a result, while it is easy to fairly compare this PPO to this TRPO, it is not the strongest PPO implementation (in the sense of sample efficiency) and can be improved on substantially.

## Supported Platforms

This package has been tested on Mac OS Mojave and Ubuntu 16.04 LTS, and is probably fine for most recent Mac and Linux operating systems. 

Requires **Python 3.6 or greater.**  

## Installation

To install this package:

```
git clone https://github.com/openai/safety-starter-agents.git

cd safety-starter-agents

pip install -e .
```

**Warning:** Installing this package does **not** install Safety Gym. If you want to use the algorithms in this package to train agents on onstrained RL environments, make sure to install Safety Gym according to the instructions on the [Safety Gym repo](https://www.github.com/openai/safety-gym).


## Getting Started

**Example Script:** To run PPO-Lagrangian on the `Safexp-PointGoal1-v0` environment from Safety Gym, using neural networks of size (64,64):

```
from safe_rl import ppo_lagrangian
import gym, safety_gym

ppo_lagrangian(
	env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
	ac_kwargs = dict(hidden_sizes=(64,64))
	)

```


**Reproduce Experiments from Paper:** To reproduce an experiment from the paper, run:

```
cd /path/to/safety-starter-agents/scripts
python experiment.py --algo ALGO --task TASK --robot ROBOT --seed SEED 
	--exp_name EXP_NAME --cpu CPU
```

where 

* `ALGO` is in `['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']`.
* `TASK` is in `['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']` .
* `ROBOT` is in `['point', 'car', 'doggo']`.
* `SEED` is an integer. In the paper experiments, we used seeds of 0, 10, and 20, but results may not reproduce perfectly deterministically across machines.
* `CPU` is an integer for how many CPUs to parallelize across.

`EXP_NAME` is an optional argument for the name of the folder where results will be saved. The save folder will be placed in `/path/to/safety-starter-agents/data`. 


**Plot Results:** Plot results with:

```
cd /path/to/safety-starter-agents/scripts
python plot.py data/path/to/experiment
```

**Watch Trained Policies:** Test policies with:

```
cd /path/to/safety-starter-agents/scripts
python test_policy.py data/path/to/experiment
```


## Cite the Paper

If you use Safety Starter Agents code in your paper, please cite:

```
@article{Ray2019,
    author = {Ray, Alex and Achiam, Joshua and Amodei, Dario},
    title = {{Benchmarking Safe Exploration in Deep Reinforcement Learning}},
    year = {2019}
}
```