# RL-Safety-Algorithms

Algorithms for Safe Reinforcement Learning Problems that were tested and 
benchmarked in the 
[Bullet-Safety-Gym](https://github.com/svengronauer/Bullet-Safety-Gym).

## Installation

Install this repository with:

```
git clone https://github.com/SvenGronauer/RL-Safety-Algorithms.git

cd RL-Safety-Algorithms

pip install -e .
```


## Getting Started

Works with every environment that is compatible with the OpenAI Gym interface:

```
python -m rl_safety_algorithms.train --alg trpo --env MountainCarContinuous-v0
```

For an open-source framework to benchmark and test safety, we recommend the 
[Bullet-Safety-Gym](https://github.com/svengronauer/Bullet-Safety-Gym). To train an
algorithms such as Constrained Policy Optimization, run:

```
python -m rl_safety_algorithms.train --alg cpo --env SafetyBallCircle-v0
```

## Benchmark

In order to benchmark tasks from the 
[Bullet-Safety-Gym](https://github.com/svengronauer/Bullet-Safety-Gym),
we have prepared scripts in the `experiments` directory.

```
cd experiments/
python benchmark_circle_tasks.py
```

In our experiments, we used a Threadripper 3990X CPU with 64 physical CPU cores,
thus, we ran the experiments with the following flag for optimal MPI usage:

```
python benchmark_circle_tasks.py --num-cores 64
```

Plots from experiment runs can be also taken from the
[Bullet-Safety-Gym Benchmarks](https://github.com/SvenGronauer/Bullet-Safety-Gym/blob/master/docs/benchmark.md)