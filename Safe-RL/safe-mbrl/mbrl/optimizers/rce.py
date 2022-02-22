'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 01:03:52
@LastEditTime: 2020-07-29 23:44:24
@Description:
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time
import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt

from .optimizer import Optimizer


class RCEOptimizer(Optimizer):
    """A Pytorch-compatible RCE optimizer.
    """
    def __init__(self, sol_dim, upper_bound, lower_bound, popsize, minimal_elites,
        max_iters=10, num_elites=20, epsilon=0.001, alpha=0.25, init_mean=0, init_var=1.5):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = torch.FloatTensor(upper_bound), torch.FloatTensor(lower_bound) # (sol_dim)
        self.epsilon, self.alpha = epsilon, alpha
        self.minimal_elites = minimal_elites

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.init_mean, self.init_var = np.tile(init_mean, self.sol_dim), np.tile(init_var, self.sol_dim) # (sol_dim)
        self.cost_function = None

    def setup(self, cost_function):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        self.cost_function = cost_function

        def sample_truncated_normal(shape, mu, sigma, a, b):
            '''
            Pytorch implementation of truncated normal distribution sampler

            Parameters:
            ----------
                @param numpy array or list - shape : size should be (popsize x sol_dim)
                @param numpy array or list - mu, sigma : size should be (sol_dim)
                @param tensor - a, b : lower bound and upper bound of sampling range, size should be (sol_dim)

            Return:
            ----------
                @param tensor - x : size should be (popsize x sol_dim)
            '''
            uniform = torch.rand(shape)
            normal = torch.distributions.normal.Normal(0, 1)

            alpha = (a - mu) / sigma
            beta = (b - mu) / sigma

            alpha_normal_cdf = normal.cdf(alpha)
            p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform

            p = p.numpy()
            one = np.array(1, dtype=p.dtype)
            epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
            v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
            x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
            #x = torch.max(torch.min(x, b), a) # sometimes the sampler may violate the upper&lower bound, so use this to make sure it will not
            return x
        self.sample_trunc_norm = sample_truncated_normal

    def reset(self):
        pass

    def obtain_solution(self, use_pytorch=True, debug=False):
        """
        Optimizes the cost function using the provided initial candidate distribution parameters

        Parameters:
        ----------
            @param numpy array - init_mean, init_var: size should be (popsize x sol_dim)
            @param bool - use_pytorch: determine if use pytorch implementation
            @param bool - debug: if true, it will save some figures to help you find the best parameters

        Return:
        ----------
            @param numpy array - sol : size should be (sol_dim)
        """

        mean, var, t = self.init_mean, self.init_var, 0

        size = [self.popsize, self.sol_dim]

        if debug:
            cost_list = []
            mean_list = []
            var_list = []

        while (t < self.max_iters) and np.max(var) > self.epsilon:

            #lb_dist, ub_dist = mean - self.lb, self.ub - mean
            #constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            mu = torch.FloatTensor(mean)
            sigma = torch.sqrt(torch.FloatTensor(var))
            samples = self.sample_trunc_norm(size, mu, sigma, self.lb, self.ub).numpy()


            cost_rewards, cost_constraints = self.cost_function(samples)
            feasible_idx = cost_constraints==0
            feasible_samples_reward = cost_rewards[feasible_idx]
            feasible_samples = samples[feasible_idx] # [num, sol_dim]
            feasible_num = feasible_samples.shape[0]
            if feasible_num<self.minimal_elites:
                idx = np.argsort(cost_constraints)
                n = self.minimal_elites - feasible_num
                sub_elites = samples[idx][:n]
                elites = np.concatenate((sub_elites, feasible_samples), axis=0)
            else:
                idx = np.argsort(feasible_samples_reward)
                elites = feasible_samples[idx][:self.num_elites]
            #print(np.sort(costs)[:self.num_elites])

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            if debug:
                min_cost = costs[idx][:self.num_elites]
                cost_list.append(np.mean(min_cost))
                mean_list.append(np.mean(new_mean[0]))
                var_list.append(np.mean(new_var))

            t += 1
            sol, solvar = mean, var

        if debug:
            fig, axs = plt.subplots(3, sharex=True)
            axs[0].plot(cost_list)
            axs[1].plot(mean_list)
            axs[2].plot(var_list)
            name = time.time()
            name = str(name)
            plt.savefig("./data/debug/"+name+".png")
            plt.close()
        return sol, solvar
