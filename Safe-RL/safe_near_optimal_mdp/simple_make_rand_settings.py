from __future__ import division, print_function, absolute_import

import GPy
import numpy as np
import arguments

from safemdp.grid_world import (draw_gp_sample, compute_S_hat0)


args = arguments.safemdp_argparse()

# Define world
world_shape = args.world_shape
step_size = args.step_size

# Define GP for safety
noise_safety = args.noise_safety
safety_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.),
                             variance=1., ARD=True)
safety_lik = GPy.likelihoods.Gaussian(variance=noise_safety ** 2)
safety_lik.constrain_bounded(1e-6, 10000.)

# Define GP for reward
noise_reward = args.noise_reward
reward_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.),
                             variance=1., ARD=True)
reward_lik = GPy.likelihoods.Gaussian(variance=noise_reward ** 2)
reward_lik.constrain_bounded(1e-6, 10000.)

# Safety and Reward functions
safety, _ = draw_gp_sample(safety_kernel, world_shape, step_size)
reward, _ = draw_gp_sample(reward_kernel, world_shape, step_size)

# Set the minimum value for reward as zero
reward -= min(reward)

# Safety threhsold, Lipschitz constant, scaling factors for confidence interval
h = args.h

# Initialize safe sets
S0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
S0[:, 0] = True
S_hat0 = compute_S_hat0(np.nan, world_shape, 4, safety, step_size, h)
start_pos = np.random.choice(np.where(S_hat0)[0])

# Save the problem settings as a npz file
np.savez('data/simple/random_settings_new', safety=safety, reward=reward,
         start_pos=start_pos)
