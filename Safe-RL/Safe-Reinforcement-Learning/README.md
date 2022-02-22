# Safe reninforcement learning:
  * **Key question**: How to design an efficient and safe policy for an autonomous vehicle?
  
## Papers:

* [Safe Model-based Reinforcement Learning with Stability Guarantees](https://arxiv.org/pdf/1705.08551.pdf)
   * **Key idea**: This paper introduces a novel algorithm that can safely optimize policies in continuous
state-action spaces while providing high-probability safety guarantees in terms of stability. Moreover,
it shows that it is possible to exploit the regularity properties of the system in order to safely learn
about the dynamics and thus improve the policy and increase the estimated safe region of attraction
without ever leaving it. Specifically, starting from a policy that is known to stabilize the system
locally, it gathers data at informative, safe points and improve the policy safely based on the improved
model of the system and prove that any exploration algorithm that gathers data at these points reaches
a natural notion of full exploration.

  * [Video](https://www.youtube.com/watch?v=Xwu38vQb9Gk)
  
  * [Slides](http://wrai.org/slides/Safe%20Reinforcement%20Learning%20in%20Robotics%20with%20Bayesian%20Models%20-%20Felix%20Berkenkamp.pdf)

* [Safe Learning of Regions of Attraction for Uncertain, Nonlinear Systems with Gaussian Processes](https://arxiv.org/pdf/1603.04915.pdf)
   * **Key idea**: This paper considers an approach that learns the region of attraction (ROA) from experiments on a real system, without ever leaving the true ROA and, thus, without risking safety-critical failures. Based on regularity assumptions on the model errors in terms of a Gaussian process prior, it uses an underlying Lyapunov function in order to determine a region in which an equilibrium point is asymptotically stable with high probability. Moreover, it provides an algorithm to actively and safely explore the state space in order to expand the ROA estimate.
   
  * [Video](https://www.youtube.com/watch?v=bSv-pNOWn7c)
  
   
* [A Lyapunov-based Approach to Safe Reinforcement Learning](https://arxiv.org/pdf/1805.07708.pdf)
   * **Key idea**: This paper derives algorithms under the framework of constrained Markov decision problems (CMDPs), an extension of the standard Markov decision problems (MDPs) augmented with constraints on expected cumulative costs. The approach is based on a novel Lyapunov method. It defines and presents a method for constructing Lyapunov functions, which provide an effective way to guarantee the global safety of a behavior policy during training via a set of local, linear constraints.
   
* [Safe Exploration for Optimization with Gaussian Processes](http://proceedings.mlr.press/v37/sui15.pdf)
   * **Key idea**: This paper models a novel class of safe optimization problems as maximizing an unknown expected reward function over the decision set from noisy samples. By exploiting regularity assumptions on the function, which capture the   intuition that similar decisions are associated with similar rewards, the gaol is to balance exploration
(learning about the function) and exploitation (identifying near-optimal decisions), while additionally ensuring safety
throughout the process. 
   
* [Learning-based Model Predictive Control for Safe Exploration and Reinforcement Learning](https://arxiv.org/pdf/1803.08287.pdf)
   * **Key idea**: This paper combines ideas from robust control and GP-based RL to design a MPC scheme that recursively   guarantees the existence of a safety trajectory that satisfies the constraints of the system. Particularly, it uses a novel uncertainty propagation technique that can reliably propagate the confidence intervals of a GP-model forward in time.
   
* [Safe Exploration in Finite Markov Decision Processes with Gaussian Processes](https://arxiv.org/pdf/1606.04753.pdf)
   * **Key idea**: This paper defines safety in terms of an, a priori unknown, safety constraint that depends on states and actions. The algorithm cautiously explores safe states and actions in order to gain statistical confidence about the safety of unvisited state-action pairs from noisy observations collected while navigating the environment. Moreover, the algorithm explicitly considers reachability when exploring the MDP, ensuring that it does not get stuck in any state with no safe way out.
   * [Video](https://www.youtube.com/watch?v=dHHh7CZQM_M)
   
* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/pdf/1610.03295.pdf)
 
   * [Video](https://www.youtube.com/watch?v=cYTVXfIH0MU&t=1784s)
   
## Survey papers:

* [Planning and Decision-Making for Autonomous Vehicles](https://www.annualreviews.org/doi/pdf/10.1146/annurev-control-060117-105157)

* [A Comprehensive Survey on Safe Reinforcement Learning](http://jmlr.org/papers/volume16/garcia15a/garcia15a.pdf)

## Lectures:

* [Safe Reinforcement Learning - Stanford CS234](https://web.stanford.edu/class/cs234/slides/2017/cs234_guest_lecture_safe_rl.pdf)

* [High Confidence Off-Policy Evaluation (HCOPE) - CMU 15-889e](http://www.cs.cmu.edu/~ebrun/15889e/lectures/thomas_lecture1_2.pdf)

## Ideas:

* **Virtual to real learnig**: Agent can learn a safe policy in a simulator (which is always safe), then transfer that policy into a real world.

  * Challenges:
    * How to deal with simulation mismatch?
      * Idea: Transfer learning comunity may provide some solution.
        * [sim 2 real](https://katefvision.github.io/katefSlides/sim2real.pdf)
        * [Transfer from Simulation to Real World through
Learning Deep Inverse Dynamics Model](https://arxiv.org/pdf/1610.03518.pdf)
        * [Using Simulation to Improve Sample-Efficiency of Bayesian
Optimization for Bipedal Robots](https://arxiv.org/pdf/1805.02732.pdf)

    * How to design a dafe controller?
        
      * Idea: [Verifying Controllers Against Adversarial Examples with Bayesian Optimization](https://arxiv.org/pdf/1802.08678.pdf)
      
 * Transfer from Simulation to Real World
  
 Some defintion:
   * Forward dynamics: maps current state and action to a next state.
   * Inverse dynamics: maps current and next state to an action that achieves the transition between the two.
    


   

 
