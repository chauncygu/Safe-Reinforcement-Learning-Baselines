## Safe-Reinforcement-Learning-Baseline




The repository is for Safe Reinforcement Learning (RL) research, in which we investigate various safe RL baselines and safe RL benchmarks, including single agent RL and multi-agent RL. If any authors do not want their paper to be listed here, please feel free to contact <gshangd[AT]foxmail.com>. (This repository is under actively development. We appreciate any constructive comments and suggestions)


You are more than welcome to update this list! If you find a paper about Safe RL which is not listed here, please

- fork this repository, add it and merge back;
- or report an issue here;
- or email <gshangd[AT]foxmail.com>.




### 1. Environments Supported:
#### 1.1. Safe Single Agent RL benchmarks:
- [Safety-Gym](https://github.com/openai/safety-gym)
- [Bullet-Safety-Gym](https://github.com/svengronauer/Bullet-Safety-Gym)
#### 1.2. Safe Multi-Agent RL benchmark:
- [Safe Multi-Agent Mujoco](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)
<!--- [Safe Multi-Robot Robosuite](https://github.com/chauncygu/Safe-Multi-Agent-Mujoco)-->


### 2. Safe RL Baselines:

#### 2.1. Safe Single Agent RL Baselines:

- Consideration of risk in reinforcement learning, [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.45.8264&rep=rep1&type=pdf), Not Find Code, (Accepted by ICML 1994)
- Multi-criteria Reinforcement Learning,  [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.962&rep=rep1&type=pdf), Not Find Code, (Accepted by ICML 1998)
- Lyapunov design for safe reinforcement learning, [Paper](https://www.jmlr.org/papers/volume3/perkins02a/perkins02a.pdf), Not Find Code, (Accepted by ICML 2002)
- Risk-sensitive reinforcement learning, [Paper](https://link.springer.com/content/pdf/10.1023/A:1017940631555.pdf), Not Find Code, (Accepted by Machine Learning, 2002)
- Risk-Sensitive Reinforcement Learning Applied to Control under Constraints, [Paper](https://www.jair.org/index.php/jair/article/view/10415/24966), Not Find Code, (Accepted by Journal of Artificial Intelligence Research, 2005)
- An actor-critic algorithm for constrained markov decision processes, [Paper](https://reader.elsevier.com/reader/sd/pii/S0167691104001276?token=D2FDE94E441EB4182DF4CF382458FCA57BDCABECB2E17932BF52CABA7F46F0F67EE5E9A4BE19F9FD3E27D4099CA25C80&originRegion=eu-west-1&originCreation=20220304073259), Not Find Code, (Accepted by Systems & Control Letters, 2005)
- Reinforcement learning for MDPs with constraints, [Paper](https://link.springer.com/content/pdf/10.1007/11871842_63.pdf), Not Find Code, (Accepted by European Conference on Machine Learning 2006)
- Discounted Markov decision processes with utility constraints, [Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.140.1315&rep=rep1&type=pdf), Not Find Code, (Accepted by Computers & Mathematics with Applications, 2006)
- Constrained reinforcement learning from intrinsic and extrinsic rewards, [Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1059.1383&rep=rep1&type=pdf), Not Find Code, (Accepted by International Conference on Development and Learning 2007)
- Safe exploration for reinforcement learning, [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.2786&rep=rep1&type=pdf), Not Find Code, (Accepted by ESANN 2008)
- Percentile optimization for Markov decision processes with parameter uncertainty, [Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.400.5048&rep=rep1&type=pdf), Not Find Code, (Accepted by Operations research, 2010)
- Safe reinforcement learning in high-risk tasks through policy improvement, [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5967356), Not Find Code, (Accepted by IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning (ADPRL) 2011) 
- Safe Exploration in Markov Decision Processes, [Paper](https://arxiv.org/pdf/1205.4810.pdf), Not Find Code, (Accepted by ICML 2012)
- Policy gradients with variance related risk criteria, [Paper](https://arxiv.org/pdf/1206.6404.pdf), Not Find Code, (Accepted by ICML 2012)
- Risk aversion in Markov decision processes via near optimal Chernoff bounds, [Paper](https://proceedings.neurips.cc/paper/2012/file/e2f374c3418c50bc30d67d5f7454a5b4-Paper.pdf), Not Find Code, (Accepted by NeurIPS 2012)
- Safe Exploration of State and Action Spaces in Reinforcement Learning, [Paper](https://web.archive.org/web/20180423223542id_/http://www.jair.org/media/3761/live-3761-6687-jair.pdf), Not Find Code, (Accepted by Journal of Artificial Intelligence Research, 2012)
- An Online Actor–Critic Algorithm with Function Approximation for Constrained Markov Decision Processes, [Paper](https://link.springer.com/content/pdf/10.1007/s10957-012-9989-5.pdf), Not Find Code, (Accepted by Journal of Optimization Theory and Applications, 2012)
- Safe policy iteration, [Paper](http://proceedings.mlr.press/v28/pirotta13.pdf), Not Find Code, (Accepted by ICML 2013)
- Safe Policy Search for Lifelong Reinforcement Learning with Sublinear Regret, [Paper](https://arxiv.org/pdf/1505.05798.pdf), Not Find Code, (Accepted by ICML 2015)
- High-Confidence Off-Policy Evaluation, [Paper](https://www.ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/2015Thomas2015.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safeRL) (Accepted by AAAI 2015)
- Safe Exploration for Optimization with Gaussian Processes, [Paper](http://proceedings.mlr.press/v37/sui15.pdf), Not Find Code (Accepted by ICML 2015)
- Safe Exploration in Finite Markov Decision Processes with Gaussian Processes, [Paper](https://proceedings.neurips.cc/paper/2016/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf), Not Find Code (Accepted by NeurIPS 2016)
- Safe and efficient off-policy reinforcement learning, [Paper](https://www.researchgate.net/profile/Anna-Harutyunyan-3/publication/303859091_Safe_and_Efficient_Off-Policy_Reinforcement_Learning/links/57b2e8c908aeb2cf17c73ad2/Safe-and-Efficient-Off-Policy-Reinforcement-Learning.pdf), [Code](https://github.com/ALRhub/Retrace-PyTorch) (Accepted by NeurIPS 2016)
- Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving, [Paper](https://arxiv.org/pdf/1610.03295.pdf?ref=https://githubhelp.com), Not Find Code (only Arxiv, 2016, citation 530+)
- Safe Learning of Regions of Attraction in Uncertain, Nonlinear Systems with Gaussian Processes, [Paper](https://arxiv.org/pdf/1603.04915.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_learning) (Accepetd by CDC 2016)
- Safety-constrained reinforcement learning for MDPs, [Paper](https://www.researchgate.net/profile/Nils-Jansen-2/publication/283118102_Safety-Constrained_Reinforcement_Learning_for_MDPs/links/5630d2af08aef3349c29f90f/Safety-Constrained-Reinforcement-Learning-for-MDPs.pdf), Not Find Code (Accepted by InInternational Conference on Tools and Algorithms for the Construction and Analysis of Systems 2016)
- Convex synthesis of randomized policies for controlled Markov chains with density safety upper bound constraints, [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7526658), Not Find Code (Accepted by American Control Conference 2016)
- Combating Deep Reinforcement Learning's Sisyphean Curse with Intrinsic Fear, [Paper](https://openreview.net/pdf?id=r1tHvHKge), Not Find Code (only Openreview, 2016)
- Constrained Policy Optimization (CPO), [Paper](http://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safety-starter-agents) (Accepted by ICML 2017)
- Risk-constrained reinforcement learning with percentile risk criteria, [Paper](https://www.jmlr.org/papers/volume18/15-636/15-636.pdf), , Not Find Code (Accepted by The Journal of Machine Learning Research, 2017)
- Probabilistically Safe Policy Transfer, [Paper](https://arxiv.org/pdf/1705.05394.pdf),  Not Find Code (Accepted by ICRA 2017) 
- Accelerated primal-dual policy optimization for safe reinforcement learning, [Paper](https://arxiv.org/pdf/1802.06480.pdf), Not Find Code (Arxiv, 2017)
- Stagewise safe bayesian optimization with gaussian processes, [Paper](http://www.yisongyue.com/publications/icml2018_stageopt.pdf),  Not Find Code (Accepted by ICML 2018)
- Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning, [Paper](https://arxiv.org/pdf/1711.06782.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/LeaveNoTrace) (Accepted by ICLR 2018)
- Safe Model-based Reinforcement Learning with Stability Guarantees, [Paper](https://proceedings.neurips.cc/paper/2017/file/766ebcd59621e305170616ba3d3dac32-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_learning) (Accepted by NeurIPS 2018)
- A Lyapunov-based Approach to Safe Reinforcement Learning, [Paper](https://proceedings.neurips.cc/paper/2018/file/4fe5149039b52765bde64beb9f674940-Paper.pdf), Not Find Code (Accepted by NeurIPS 2018)
- Constrained Cross-Entropy Method for Safe Reinforcement Learning, [Paper](https://proceedings.neurips.cc/paper/2018/file/34ffeb359a192eb8174b6854643cc046-Paper.pdf), Not Find Code (Accepted by NeurIPS 2018)
- Safe Reinforcement Learning via Formal Methods, [Paper](http://www.cs.cmu.edu/~aplatzer/pub/SafeRL.pdf), Not Find Code (Accepted by AAAI 2018)
- Safe reinforcement learning via shielding, [Paper](https://arxiv.org/pdf/1708.08611.pdf), [Code](https://github.com/safe-rl/safe-rl-shielding) (Accepted by AAAI 2018)
- Trial without Error: Towards Safe Reinforcement Learning via Human Intervention, [Paper](https://www.ifaamas.org/Proceedings/aamas2018/pdfs/p2067.pdf), Not Find Code (Accepted by AAMAS 2018)
- Learning-based Model Predictive Control for Safe Exploration and Reinforcement Learning, [Paper](https://arxiv.org/pdf/1906.12189.pdf), Not Find Code (Accepted by CDC 2018)
- The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems, [Paper](http://proceedings.mlr.press/v87/richards18a/richards18a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_learning) (Accepted by CoRL 2018)
- OptLayer - Practical Constrained Optimization for Deep Reinforcement Learning in the Real World, [Paper](https://arxiv.org/pdf/1709.07643.pdf), Not Find Code (Accepted by ICRA 2018)
- Safe reinforcement learning on autonomous vehicles, [Paper](https://arxiv.org/pdf/1910.00399.pdf), Not Find Code (Accepted by IROS 2018)
- Safe reinforcement learning: Learning with supervision using a constraint-admissible set, [Paper](https://ieeexplore.ieee.org/abstract/document/8430770), Not Find Code (Accepted by Annual American Control Conference (ACC) 2018)
- Safe Exploration in Continuous Action Spaces, [Paper](https://www.researchgate.net/profile/Gal-Dalal/publication/322756278_Safe_Exploration_in_Continuous_Action_Spaces/links/5a71e84faca2720bc0d940b3/Safe-Exploration-in-Continuous-Action-Spaces.pdf), [Code](https://github.com/AgrawalAmey/safe-explorer), (only Arxiv, 2018, citation 200+)
- Safe exploration of nonlinear dynamical systems: A predictive safety filter for reinforcement learning, [Paper](https://www.researchgate.net/profile/Kim-Wabersich/publication/329641554_Safe_exploration_of_nonlinear_dynamical_systems_A_predictive_safety_filter_for_reinforcement_learning/links/5ede2aab299bf1d20bd87981/Safe-exploration-of-nonlinear-dynamical-systems-A-predictive-safety-filter-for-reinforcement-learning.pdf), Not Find Code, (Arxiv, 2018, citation 40+)
- Batch policy learning under constraints, [Paper](http://proceedings.mlr.press/v97/le19a/le19a.pdf), [Code](https://github.com/clvoloshin/constrained_batch_policy_learning), (Accepted by ICML 2019)
- Convergent Policy Optimization for Safe Reinforcement Learning, [Paper](https://proceedings.neurips.cc/paper/2019/file/db29450c3f5e97f97846693611f98c15-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/Safe_reinforcement_learning) (Accepted by NeurIPS 2019)
- Constrained reinforcement learning has zero duality gap, [Paper](https://www.researchgate.net/profile/Luiz-Chamon/publication/336889860_Constrained_Reinforcement_Learning_Has_Zero_Duality_Gap/links/5ef4df204585155050726b42/Constrained-Reinforcement-Learning-Has-Zero-Duality-Gap.pdf), Not Find Code (Accepted by NeurIPS 2019)
- Reinforcement learning with convex constraints, [Paper](https://www.cs.princeton.edu/~syoosefi/papers/NeurIPS2019.pdf), [Code](https://github.com/xkianteb/ApproPO) (Accepted by NeurIPS 2019)
- Reward constrained policy optimization, [Paper](https://arxiv.org/pdf/1805.11074.pdf), Not Find Code (Accepted by ICLR 2019)
- Supervised policy update for deep reinforcement learning, [Paper](https://arxiv.org/pdf/1805.11706.pdf), [Code](https://github.com/quanvuong/Supervised_Policy_Update), (Accepted by ICLR 2019)
- Lyapunov-based safe policy optimization for continuous control, [Paper](https://openreview.net/pdf?id=SJgUYBVLsN), Not Find Code (Accepted by ICML Workshop RL4RealLife 2019)
- Safe reinforcement learning with model uncertainty estimates, [Paper](https://arxiv.org/pdf/1810.08700.pdf), Not Find Code (Accepted by ICRA 2019)
- Safe reinforcement learning with scene decomposition for navigating complex urban environments, [Paper](https://arxiv.org/pdf/1904.11483.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/AutomotiveSafeRL), (Accepted by IV 2019)
- Probabilistic policy reuse for safe reinforcement learning, [Paper](https://dl.acm.org/doi/pdf/10.1145/3310090?casa_token=OahWDUpVTxAAAAAA:MVJd1GjD6HDpFKMxXfp9pd3KaJbG879P7qvcMS0-VDGFAR0prYuXwzN9LwI4BfkPti085CGGhsz1llY), Not Find Code, (Accepted by ACM Transactions on Autonomous and Adaptive Systems (TAAS), 2019)
- Projected stochastic primal-dual method for constrained online learning with kernels, [Paper](https://ieeexplore.ieee.org/ielaam/78/8691646/8678800-aam.pdf), Not Find Code, (Accepted by IEEE Transactions on Signal Processing, 2019)
- Temporal logic guided safe reinforcement learning using control barrier functions, [Paper](https://arxiv.org/pdf/1903.09885.pdf), Not Find Code (Arxiv, Citation 25+, 2019)
- Safe policies for reinforcement learning via primal-dual methods, [Paper](https://www.researchgate.net/profile/Luiz-Chamon/publication/337438444_Safe_Policies_for_Reinforcement_Learning_via_Primal-Dual_Methods/links/5ef4df1f299bf18816e7f62c/Safe-Policies-for-Reinforcement-Learning-via-Primal-Dual-Methods.pdf), Not Find Code (Arxiv, Citation 25+, 2019)
- Value constrained model-free continuous control, [Paper](https://arxiv.org/pdf/1902.04623.pdf), Not Find Code (Arxiv, Citation 35+, 2019)
- Safe Reinforcement Learning in Constrained Markov Decision Processes (SNO-MDP), [Paper](http://proceedings.mlr.press/v119/wachi20a/wachi20a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe_near_optimal_mdp) (Accepted by ICML 2020)
- Responsive Safety in Reinforcement Learning by PID Lagrangian Methods, [Paper](http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf), [Code](https://github.com/keirp/glamor/tree/98681a23bae9e8e5e9fbf68a0316ca2a22a27593/dependencies/rlpyt/rlpyt/projects/safe) (Accepted by ICML 2020)
- Constrained markov decision processes via backward value functions, [Paper](http://proceedings.mlr.press/v119/satija20a/satija20a.pdf), [Code](https://github.com/hercky/cmdps_via_bvf/tree/69b9f51cb6410673d0aa2e5b9c980b33e5a46dda) (Accepted by ICML 2020)
- Projection-Based Constrained Policy Optimization (PCPO), [Paper](https://arxiv.org/pdf/2010.03152.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/PCPO) (Accepted by ICLR 2020)
- First order constrained optimization in policy space (FOCOPS),[Paper](https://proceedings.neurips.cc/paper/2020/file/af5d5ef24881f3c3049a7b9bfe74d58b-Paper.pdf), [Code](https://github.com/ymzhang01/focops) (Accepted by NeurIPS 2020)
- Safe reinforcement learning via curriculum induction, [Paper](https://proceedings.neurips.cc/paper/2020/file/8df6a65941e4c9da40a4fb899de65c55-Paper.pdf), [Code](https://github.com/zuzuba/CISR_NeurIPS20) (Accepted by NeurIPS 2020)
- Constrained episodic reinforcement learning in concave-convex and knapsack settings, [Paper](https://arxiv.org/pdf/2006.05051.pdf), [Code](https://github.com/miryoosefi/ConRL) (Accepted by NeurIPS 2020)
- IPO: Interior-point Policy Optimization under Constraints, [Paper](https://www.researchgate.net/profile/Yongshuai-Liu/publication/336735393_IPO_Interior-point_Policy_Optimization_under_Constraints/links/5e1670874585159aa4bff037/IPO-Interior-point-Policy-Optimization-under-Constraints.pdf), Not Find Code (Accepted by AAAI 2020)
- Safe reinforcement learning using robust mpc, [Paper](https://arxiv.org/pdf/1906.04005.pdf), Not Find Code (IEEE Transactions on Automatic Control, 2020)
- Safe reinforcement learning via projection on a safe set: How to achieve optimality?, [Paper](https://arxiv.org/pdf/2004.00915.pdf), Not Find Code (Accepted by IFAC 2020)
- Learning Transferable Domain Priors for Safe Exploration in Reinforcement Learning, [Paper](https://arxiv.org/pdf/1909.04307.pdf), [Code](https://github.com/GKthom/Priors-for-safe-exploration), (Accepted by International Joint Conference on Neural Networks (IJCNN) 2020)
- Learning safe policies with cost-sensitive advantage estimation, [Paper](https://openreview.net/pdf?id=uVnhiRaW3J), Not Find Code (Openreview 2020)
- Safe reinforcement learning using probabilistic shields, [Paper](https://repository.ubn.ru.nl/bitstream/handle/2066/224966/224966.pdf?sequence=1), Not Find Code (2020)
- Safe reinforcement learning using advantage-based intervention, [Paper](http://proceedings.mlr.press/v139/wagener21a/wagener21a.pdf), [Code](https://github.com/nolanwagener/safe_rl) (Accepted by ICML 2021)
- Shortest-path constrained reinforcement learning for sparse reward tasks, [Paper](https://arxiv.org/pdf/2107.06405.pdf), [Code](https://github.com/srsohn/shortest-path-rl), (Accepted by ICML 2021)
- Density constrained reinforcement learning, [Paper](https://arxiv.org/pdf/2106.12764.pdf), Not Find Code (Accepted by ICML 2021)
- Safe Reinforcement Learning by Imagining the Near Future (SMBPO), [Paper](https://proceedings.neurips.cc/paper/2021/file/73b277c11266681122132d024f53a75b-Paper.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/Safe-MBPO) (Accepted by NeurIPS 2021)
- Exponential Bellman Equation and Improved Regret Bounds for Risk-Sensitive Reinforcement Learning, [Paper](https://arxiv.org/pdf/2111.03947.pdf),  Not Find Code (Accepted by NeurIPS 2021)
- Risk-Sensitive Reinforcement Learning: Symmetry, Asymmetry, and Risk-Sample Tradeoff, [Paper](https://arxiv.org/pdf/2111.03947.pdf),  Not Find Code (Accepted by NeurIPS 2021)
- Safe reinforcement learning with natural language constraints, [Paper](https://proceedings.neurips.cc/paper/2021/file/72f67e70f6b7cdc4cc893edaddf0c4c6-Paper.pdf), [Code](https://github.com/princeton-nlp/SRL-NLC), (Accepted by NeurIPS 2021)
-  Conservative safety critics for exploration, [Paper](https://arxiv.org/pdf/2010.14497.pdf), Not Find Code (Accepted by ICLR 2021)
-  Risk-averse trust region optimization for reward-volatility reduction, [Paper](https://arxiv.org/pdf/1912.03193.pdf), Not Find Code (Accepted by IJCAI 2021)
- AlwaysSafe: Reinforcement Learning Without Safety Constraint Violations During Training, [Paper](https://pure.tudelft.nl/ws/files/96913978/p1226.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/AlwaysSafe) (Accepted by AAMAS 2021)
- Safe Continuous Control with Constrained Model-Based Policy Optimization (CMBPO), [Paper](https://arxiv.org/pdf/2104.06922.pdf), [Code](https://github.com/anyboby/Constrained-Model-Based-Policy-Optimization) (Accepted by IROS 2021)
- Context-aware safe reinforcement learning for non-stationary environments, [Paper](https://arxiv.org/pdf/2101.00531.pdf), Not Find Code (Accepted by ICRA 2021)
- Safe model-based reinforcement learning with robust cross-entropy method, [Paper](https://aisecure-workshop.github.io/aml-iclr2021/papers/8.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/safe-mbrl) (Accepted by ICLR 2021 Workshop on Security and Safety in Machine Learning Systems)
- Safe Reinforcement Learning of Control-Affine Systems with Vertex Networks, [Paper](http://proceedings.mlr.press/v144/zheng21a/zheng21a.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-RL/vertex-net) (Accepted by Conference on Learning for Dynamics and Control 2021)
- Can You Trust Your Autonomous Car? Interpretable and Verifiably Safe Reinforcement Learning, [Paper](http://download.cmutschler.de/publications/2021/IV2021.pdf), Not Find Code (Accepted by IV 2021)
- Recovery RL: Safe Reinforcement Learning with Learned Recovery Zones, [Paper](https://www.researchgate.net/profile/Minho-Hwang/publication/345152769_Recovery_RL_Safe_Reinforcement_Learning_with_Learned_Recovery_Zones/links/5fe37ea2299bf140883a35cb/Recovery-RL-Safe-Reinforcement-Learning-with-Learned-Recovery-Zones.pdf), [Code](https://github.com/abalakrishna123/recovery-rl), (Accepted by IEEE RAL, 2021)
- Reinforcement learning control of constrained dynamic systems with uniformly ultimate boundedness stability guarantee, [Paper](https://www.sciencedirect.com/science/article/pii/S0005109821002090), Not Find Code (Accepted by Automatica, 2021)
- A simple reward-free approach to constrained reinforcement learning, [Paper](https://www.cs.princeton.edu/~syoosefi/papers/reward-free2021.pdf),  Not Find Code (Arxiv, 2021)
- State augmented constrained reinforcement learning: Overcoming the limitations of learning with rewards, [Paper](https://arxiv.org/pdf/2102.11941.pdf),  Not Find Code (Arxiv, 2021)
- Safe reinforcement learning using robust action governor, [Paper](https://arxiv.org/pdf/2102.10643.pdf), Not Find Code (Accepted by In Learning for Dynamics and Control, 2022)
- A primal-dual approach to constrained markov decision processes, [Paper](https://arxiv.org/pdf/2101.10895.pdf),  Not Find Code (Arxiv, 2022)
- SAUTE RL: Almost Surely Safe Reinforcement Learning Using State Augmentation, [Paper](https://arxiv.org/pdf/2202.06558.pdf), Not Find Code (Arxiv, 2022)
- Finding Safe Zones of policies Markov Decision Processes, [Paper](https://arxiv.org/pdf/2202.11593.pdf), Not Find Code (Arxiv, 2022)
- CUP: A Conservative Update Policy Algorithm for Safe Reinforcement Learning, [Paper](https://arxiv.org/pdf/2202.07565.pdf), [Code](https://github.com/RL-boxes/Safe-RL) (Arxiv, 2022)
- SAFER: Data-Efficient and Safe Reinforcement Learning via Skill Acquisition, [Paper](https://arxiv.org/pdf/2202.04849.pdf), Not Find Code (Arxiv, 2022)



#### 2.2. Safe Multi-Agent RL Baselines:
- Multi-Agent Constrained Policy Optimisation (MACPO), [Paper](https://arxiv.org/pdf/2110.02793.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-MARL/Multi-Agent-Constrained-Policy-Optimisation) (Arxiv, 2021)
- MAPPO-Lagrangian, [Paper](https://arxiv.org/pdf/2110.02793.pdf), [Code](https://github.com/chauncygu/Safe-Reinforcement-Learning-Baseline/tree/main/Safe-MARL/Multi-Agent-Constrained-Policy-Optimisation)  (Arxiv, 2021)

#### 3. Surveys
- A comprehensive survey on safe reinforcement learning, [Paper](https://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf) (Accepted by Journal of Machine Learning Research, 2015)
- Safe learning and optimization techniques: Towards a survey of the state of the art, [Paper](https://arxiv.org/pdf/2101.09505.pdf) (Accepted by In International Workshop on the Foundations of Trustworthy AI Integrating Learning, Optimization and Reasoning, 2020)
- Safe learning in robotics: From learning-based control to safe reinforcement learning, [Paper](https://arxiv.org/pdf/2108.06266.pdf) (Accepted by Annual Review of Control, Robotics, and Autonomous Systems, 2021)
- Policy learning with constraints in model-free reinforcement learning: A survey, [Paper](https://web.archive.org/web/20210812230501id_/https://www.ijcai.org/proceedings/2021/0614.pdf) (Accepted by IJCAI 2021)

#### 4. Thesis
- Safe reinforcement learning, [Thesis](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1527&context=dissertations_2) (PhD thesis, Philip S. Thomas, University of Massachusetts Amherst, 2015)

#### 5. Book
- Constrained Markov decision processes: stochastic modeling, [Book](https://www-sop.inria.fr/members/Eitan.Altman/PAPERS/h.pdf), (Eitan Altman, Routledge, 1999)




