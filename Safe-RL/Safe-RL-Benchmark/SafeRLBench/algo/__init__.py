"""Algorithm Module.

=================== =========================================
Algorithm
=============================================================
A3C                 Asynchronous Actor-Critic Agents
PolicyGradient      Different Policy Gradient Implementations
DiscreteQLearning   Q-Learning using a table
SafeOpt             Bayesian Optimization with SafeOpt
SafeOptSwarm        Bayesion Optimization with SafeOptSwarm
=================== =========================================
"""

from .policygradient import PolicyGradient
from .safeopt import SafeOpt, SafeOptSwarm
from .a3c import A3C
from .q_learning import DiscreteQLearning

__all__ = ['PolicyGradient', 'SafeOpt', 'A3C', 'DiscreteQLearning',
           'SafeOptSwarm']
