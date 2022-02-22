'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:49:12
@LastEditTime: 2020-07-15 16:24:40
@Description:
'''

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import time
import numpy as np
import torch

from .optimizer import Optimizer


class RandomOptimizer(Optimizer):
    def __init__(self, sol_dim, upper_bound, lower_bound, popsize):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space, horizon * action dim
            popsize (int): The number of candidate solutions to be sampled at every iteration
            upper_bound (np.array, size = sol_dim): An array of upper bounds
            lower_bound (np.array, size = sol_dim): An array of lower bounds
            other parameters are not used in this optimizer
        """
        super().__init__()
        self.sol_dim = sol_dim
        self.popsize = popsize
        self.ub, self.lb = torch.FloatTensor(upper_bound), torch.FloatTensor(lower_bound) # (sol_dim)
        self.solution = None
        self.cost_function = None

    def setup(self, cost_function):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.

        Returns: None
        """
        #print("lb, ub", self.lb, self.ub)
        self.cost_function = cost_function
        self.sampler = torch.distributions.uniform.Uniform(self.lb, self.ub)
        self.size = [self.popsize]

    def reset(self):
        pass

    def obtain_solution(self, *args, **kwargs):
        """Optimizes the cost function provided in setup().
        """
   
        solutions = self.sampler.sample(self.size).cpu().numpy() # [self.popsize, self.sol_dim]
        #solutions = np.random.uniform(self.lb, self.ub, )
        costs = self.cost_function(solutions)
        
        return solutions[np.argmin(costs)], None
