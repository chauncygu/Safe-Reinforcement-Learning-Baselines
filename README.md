## Safe-Reinforcement-Learning-Baseline




The repository is for Safe Reinforcement Learning research, in which we investigate various safe RL baselines and safe RL benchmarks (including single agent RL and multi-agent RL).



### 1. Environments Supported:
#### 1.1 Safe Single Agent RL benchmarks:
- [Safety-gym](https://github.com/openai/safety-gym)
#### 1.2 Safe Multi-Agent RL benchmarks:
- [Safe Multi-Agent Mujoco](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)
- [Safe Multi-Robot Robosuite](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)


### 2. Safe RL Baselines:

#### 2.1 Safe Single Agent RL Baselines:
- Constrained Policy Optimization (CPO), [Paper](http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safety-starter-agents) (Accepted by ICML 2017)
- Projection-Based Constrained Policy Optimization (PCPO), [Paper](https://arxiv.org/pdf/2010.03152.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/PCPO) (Accepted by ICLR 2020)
- Safe Reinforcement Learning by Imagining the Near Future (SMBPO), [Paper](https://proceedings.neurips.cc/paper/2021/file/73b277c11266681122132d024f53a75b-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/Safe-MBPO) (Accepted by NeurIPS 2021)
- AlwaysSafe: Reinforcement Learning Without Safety Constraint Violations During Training, [Paper](https://pure.tudelft.nl/ws/files/96913978/p1226.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/AlwaysSafe) (Accepted by AAMAS 2021)
- Safe Continuous Control with Constrained Model-Based Policy Optimization (CMBPO), [Paper](https://arxiv.org/pdf/2104.06922.pdf), (Code)(https://github.com/anyboby/Constrained-Model-Based-Policy-Optimization) (Accepted by IROS 2021)

#### 2.2 Safe Multi-Agent RL Baselines:

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






