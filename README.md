## Safe-Reinforcement-Learning-Baseline




The repository is for Safe Reinforcement Learning research, in which we investigate various safe RL baselines and safe RL benchmarks, including single agent RL and multi-agent RL. (This repository is under actively development. We appreciated any constructive comments and suggestions)



### 1. Environments Supported:
#### 1.1 Safe Single Agent RL benchmarks:
- [Safety-Gym](https://github.com/openai/safety-gym)
- [Bullet-Safety-Gym](https://github.com/svengronauer/Bullet-Safety-Gym)
#### 1.2 Safe Multi-Agent RL benchmarks:
- [Safe Multi-Agent Mujoco](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)
- [Safe Multi-Robot Robosuite](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)


### 2. Safe RL Baselines:

#### 2.1 Safe Single Agent RL Baselines:
- High-Confidence Off-Policy Evaluation, [Paper](https://www.ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/2015Thomas2015.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safeRL) (Accepted by AAAI 2015)
- Safe Exploration for Optimization with Gaussian Processes, [Paper](http://proceedings.mlr.press/v37/sui15.pdf), , Not Find Code (Accepted by ICML 2015)
- Safe Exploration in Finite Markov Decision Processes with Gaussian Processes, [Paper](https://proceedings.neurips.cc/paper/2016/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf), Not Find Code (Accepted by NeurIPS 2016)
- Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving, [Paper](https://arxiv.org/pdf/1610.03295.pdf?ref=https://githubhelp.com), Not Find Code (only Arxiv, 2016, citation 530+)
- Safe Learning of Regions of Attraction in Uncertain, Nonlinear Systems with Gaussian Processes, [Paper](https://arxiv.org/pdf/1603.04915.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_learning) (Accepetd by CDC 2016)
- Constrained Policy Optimization (CPO), [Paper](http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safety-starter-agents) (Accepted by ICML 2017)
- Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning, [Paper](https://arxiv.org/pdf/1711.06782.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/LeaveNoTrace) (Accepted by ICLR 2018)
- Safe Model-based Reinforcement Learning with Stability Guarantees, [Paper](https://proceedings.neurips.cc/paper/2017/file/766ebcd59621e305170616ba3d3dac32-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_learning) (Accepted by NeurIPS 2018)
- A Lyapunov-based Approach to Safe Reinforcement Learning, [Paper](https://proceedings.neurips.cc/paper/2018/file/4fe5149039b52765bde64beb9f674940-Paper.pdf), Not Find Code (Accepted by NeurIPS 2018)
- Constrained Cross-Entropy Method for Safe Reinforcement Learning, [Paper](https://proceedings.neurips.cc/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf), Not Find Code (Accepted by NeurIPS 2018)
- Learning-based Model Predictive Control for Safe Exploration and Reinforcement Learning, [Paper](https://arxiv.org/pdf/1906.12189.pdf), Not Find Code (Accepted by CDC 2018)
- The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems, [Paper](http://proceedings.mlr.press/v87/richards18a/richards18a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_learning) (Accepted by CoRL 2018)
- Convergent Policy Optimization for Safe Reinforcement Learning, [Paper](https://proceedings.neurips.cc/paper/2019/file/db29450c3f5e97f97846693611f98c15-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/Safe_reinforcement_learning) (Accepted by NeurIPS 2019)
- Safe reinforcement learning with scene decomposition for navigating complex urban environments, [Paper](https://arxiv.org/pdf/1904.11483.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/AutomotiveSafeRL), (Accepted by IV 2019)
- Safe Reinforcement Learning in Constrained Markov Decision Processes (SNO-MDP), [Paper](http://proceedings.mlr.press/v119/wachi20a/wachi20a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_near_optimal_mdp) (Accepted by ICML 2020)
- Projection-Based Constrained Policy Optimization (PCPO), [Paper](https://arxiv.org/pdf/2010.03152.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/PCPO) (Accepted by ICLR 2020)
- Safe Reinforcement Learning by Imagining the Near Future (SMBPO), [Paper](https://proceedings.neurips.cc/paper/2021/file/73b277c11266681122132d024f53a75b-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/Safe-MBPO) (Accepted by NeurIPS 2021)
- AlwaysSafe: Reinforcement Learning Without Safety Constraint Violations During Training, [Paper](https://pure.tudelft.nl/ws/files/96913978/p1226.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/AlwaysSafe) (Accepted by AAMAS 2021)
- Safe Continuous Control with Constrained Model-Based Policy Optimization (CMBPO), [Paper](https://arxiv.org/pdf/2104.06922.pdf), [Code](https://github.com/anyboby/Constrained-Model-Based-Policy-Optimization) (Accepted by IROS 2021)
- Safe model-based reinforcement learning with robust cross-entropy method, [Paper](https://aisecure-workshop.github.io/aml-iclr2021/papers/8.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe-mbrl) (Accepted by ICLR 2021 Workshop on Security and Safety in Machine Learning Systems)
- Safe Reinforcement Learning of Control-Affine Systems with Vertex Networks, [Paper](http://proceedings.mlr.press/v144/zheng21a/zheng21a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/vertex-net) (Accepted by Conference on Learning for Dynamics and Control 2021)

#### 2.2 Safe Multi-Agent RL Baselines:
- Multi-Agent Constrained Policy Optimisation (MACPO), [Paper](https://arxiv.org/pdf/2110.02793.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-MARL/Multi-Agent-Constrained-Policy-Optimisation) 
- MAPPO-Lagrangian, [Paper](https://arxiv.org/pdf/2110.02793.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-MARL/Multi-Agent-Constrained-Policy-Optimisation) 


<!--
## 1. Installation

####  1.1 Create Environment

``` Bash
# create conda environment
conda create -n macpo python==3.7
conda activate macpo
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

```
cd MACPO/macpo (for the macpo algorithm) or cd MAPPO-Lagrangian/mappo_lagrangian (for the mappo_lagrangian algorithm)
pip install -e .
```



#### 1.2 Install Safety Multi-Agent Mujoco


- Install mujoco accoring to [mujoco-py](https://github.com/openai/mujoco-py) and [MuJoCo website](https://www.roboti.us/license.html).
- clone [Safety Multi-Agent Mujoco](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco) to the env path (in this repository, have set the path).

``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco200/bin;
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```



## 2. Train

```
cd MACPO/macpo/scripts or cd MAPPO-Lagrangian/mappo_lagrangian/scripts
chmod +x ./train_mujoco.sh
./train_mujoco.sh
```





## Acknowledgments

Some sections of the repository are based on [MAPPO](https://github.com/marlbenchmark/on-policy), [HAPPO](https://github.com/cyanrain7/Trust-Region-Policy-Optimisation-in-Multi-Agent-Reinforcement-Learning), [safety-starter-agents](https://github.com/openai/safety-starter-agents), [CMBPO](https://github.com/anyboby/Constrained-Model-Based-Policy-Optimization).

--!>




