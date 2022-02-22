#!/usr/bin/env python

import numpy as np
import GPy

import arguments

from safemdp.grid_world import draw_gp_sample
from gp_safety_gym import Engine_GP

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
safety_map, _ = draw_gp_sample(safety_kernel, world_shape, step_size)
reward_map, _ = draw_gp_sample(reward_kernel, world_shape, step_size)

# Set the minimum value for reward as zero
reward_map -= min(reward_map)

config = {
    'robot_base': 'xmls/point.xml',
    'task': 'goal',
    'goal_size': 0.0001,  # Set to very small value
    'robot_locations': [[3, 3], world_shape]
}

env = Engine_GP(config, reward_map, safety_map, world_shape)
obs = env.reset()
done = False

while True:
    if done:
        obs = env.reset()
    assert env.observation_space.contains(obs)

    pos = np.random.uniform(-3.5, 3.5, 2)
    print(pos)
    env.discreate_move(pos)
