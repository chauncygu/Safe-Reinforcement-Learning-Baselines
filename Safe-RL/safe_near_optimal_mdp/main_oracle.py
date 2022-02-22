from __future__ import division, print_function, absolute_import

import GPy
import numpy as np

import arguments

from safemdp.grid_world import (compute_true_safe_set, compute_S_hat0,
                                compute_true_S_hat)

from utils.safety_utilities import SafetyObj
from utils.mdp_utilities import (calc_opt_policy, reward_oracle)


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
npz_settings = np.load('data/simple/random_settings.npz')
safety = npz_settings['safety']
reward = npz_settings['reward']

# Define coordinates
n, m = world_shape
step1, step2 = step_size
xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                     np.linspace(0, (m - 1) * step2, m),
                     indexing="ij")
coord = np.vstack((xx.flatten(), yy.flatten())).T

# Safety threhsold
h = args.h
# Lipschitz constant
L = args.L
# Scaling factors for confidence intervals
beta_safety = args.beta_safety
beta_reward = args.beta_reward

# Data to initialize GP for safety and reward
n_samples = args.n_samples
ind = np.random.choice(range(safety.size), n_samples)
X = coord[ind, :]

Y_safety = safety[ind].reshape(n_samples, 1) + np.random.randn(n_samples, 1)
Y_reward = reward[ind].reshape(n_samples, 1) + np.random.randn(n_samples, 1)

# Initialize safe sets
S0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
S0[:, 0] = True
S_hat0 = compute_S_hat0(np.nan, world_shape, 4, safety, step_size, h)

# Define GP for safety and reward
gp_safety = GPy.core.GP(X, Y_safety, safety_kernel, safety_lik)

# Define Pessimistic Safety object
oracle_x = SafetyObj(gp_safety, world_shape, step_size, beta_safety,
                     safety, h, S0, S_hat0, L, update_dist=0)

# Insert samples from (s, a) in S_hat0
tmp = np.arange(oracle_x.coord.shape[0])
s_vec_ind = np.random.choice(tmp[np.any(oracle_x.S_hat[:, 1:], axis=1)])
tmp = np.arange(1, oracle_x.S.shape[1])
actions = tmp[oracle_x.S_hat[s_vec_ind, 1:].squeeze()]
for i in range(3):
    oracle_x.add_observation(s_vec_ind, np.random.choice(actions))

# Remove samples used for GP initialization
oracle_x.gp.set_XY(oracle_x.gp.X[n_samples:, :], oracle_x.gp.Y[n_samples:])

history_reward = np.zeros(args.max_time_steps)

true_S = compute_true_safe_set(oracle_x.world_shape, safety, oracle_x.h)
true_S_hat = compute_true_S_hat(oracle_x.graph, true_S, oracle_x.initial_nodes)

# Initial position
pos = npz_settings['start_pos']

# Initialization of GP-Safety-Gym
if args.render_gym:
    from gp_safety_gym import (Engine_GP, coord_to_safety_gym_pos)
    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'goal_size': 0.0001,  # Set to very small value
        'robot_locations': [coord_to_safety_gym_pos(coord[pos], world_shape)]
        }
    env = Engine_GP(config, reward_map=reward, safety_map=safety,
                    world_shape=world_shape)
    obs = env.reset()
    done = False

# Main loop
for i in range(args.max_time_steps):
    oracle_x.S_hat = true_S_hat.copy()
    # reward for oracle agent
    r_oracle = reward_oracle(oracle_x, reward)

    # Calculate the optimal policy
    next_sample, V, pi, P = calc_opt_policy(oracle_x, r_oracle, pos,
                                            args, bin_tran=True)
    # Move to the next position
    pos = next_sample[0]
    # Reward history
    history_reward[i] = reward[pos]

    if i % 50 == 0:
        print(i, pos, reward[pos])

    # Visualize using GP-Safety-Gym
    if args.render_gym:
        pos_safety_gym = coord_to_safety_gym_pos(coord[pos], world_shape)
        env.discreate_move(pos_safety_gym)

if args.render_gym:
    env.close()

file_name = "result/oracle"
np.savez(file_name, history_reward=history_reward)
