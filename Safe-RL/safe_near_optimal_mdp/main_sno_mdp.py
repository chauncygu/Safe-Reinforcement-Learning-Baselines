from __future__ import division, print_function, absolute_import

import GPy
import numpy as np

import arguments

from safemdp.grid_world import (compute_S_hat0, shortest_path)

from utils.reward_utilities import RewardObj
from utils.safety_utilities import SafetyObj
from utils.mdp_utilities import (calc_opt_policy, reward_w_exp_bonus,
                                 check_safe_exp_exit_cond)



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
# Scaling factors for confidence interval
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
gp_reward = GPy.core.GP(X, Y_reward, reward_kernel, reward_lik)

# Define Pessimistic Safety object
pes_safe_x = SafetyObj(gp_safety, world_shape, step_size, beta_safety,
                       safety, h, S0, S_hat0, L, update_dist=0)

# Define Optimistic Safety object
opt_safe_x = SafetyObj(gp_safety, world_shape, step_size, beta_safety,
                       safety, h, S0, S_hat0, L, update_dist=0)

# Define Reward object
reward_x = RewardObj(gp_reward, beta_reward)

# Insert samples from (s, a) in S_hat0
tmp = np.arange(pes_safe_x.coord.shape[0])
s_vec_ind = np.random.choice(tmp[np.any(pes_safe_x.S_hat[:, 1:], axis=1)])
tmp = np.arange(1, pes_safe_x.S.shape[1])
actions = tmp[pes_safe_x.S_hat[s_vec_ind, 1:].squeeze()]
for i in range(3):
    pes_safe_x.add_observation(s_vec_ind, np.random.choice(actions))
    opt_safe_x.add_observation(s_vec_ind, np.random.choice(actions))

# Remove samples used for GP initialization
pes_safe_x.gp.set_XY(pes_safe_x.gp.X[n_samples:, :],
                     pes_safe_x.gp.Y[n_samples:])

opt_safe_x.gp.set_XY(pes_safe_x.gp.X[n_samples:, :],
                     pes_safe_x.gp.Y[n_samples:])

history_reward = np.zeros(args.max_time_steps)
rec_point = 0

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

# [STEP 1] Exploration of Safety
for i in range(args.max_iter_safe_exp):
    # Update optimistic and pessimistic safe space
    pes_safe_x.update_sets()
    opt_safe_x.update_opt_sets()
    opt_safe_x.S_bar |= pes_safe_x.S_hat

    # Check whether or not exploration of safety can be finished
    if i % args.freq_check == 0:
        if args.es2_type == 'es2':
            exit_flag = check_safe_exp_exit_cond(pes_safe_x, opt_safe_x,
                                                 reward_x, args, bin_tran=True)
        elif args.es2_type == 'p_es2':
            exit_flag = check_safe_exp_exit_cond(pes_safe_x, opt_safe_x,
                                                 reward_x, args, bin_tran=False)
        else:
            exit_flag = False

    if exit_flag:
        print("\n Exploration of safety ended (Early Stopping) STEP ", i, "\n")
        if i == 0:
            next_sample = pes_safe_x.target_sample()
        break

    # Select the next sample
    next_sample = pes_safe_x.target_sample()

    # Set the initial position to the firstly sampled state
    if i == 0:
        cur_pos = next_sample[0]

    # Calculate the path to the next sampled point
    path = shortest_path(cur_pos, next_sample, pes_safe_x.graph)

    # Reward_history
    for idx_j, j in enumerate(path):
        history_reward[rec_point+idx_j] = reward[j]
    rec_point += idx_j+1

    # Update the current position
    cur_pos = next_sample[0]

    # Update GP for safety and reward
    pes_safe_x.add_observation(*next_sample)
    opt_safe_x.add_observation(*next_sample)
    reward_x.add_gp_observations(coord[cur_pos, :], reward[cur_pos])

    # Add GP observations for neighboring state
    if args.multi_obs:
        for _, nei, act in pes_safe_x.graph.edges_iter(cur_pos, data='action'):
            pes_safe_x.add_observation(nei, act)
            opt_safe_x.add_observation(nei, act)
            reward_x.add_gp_observations(coord[nei, :], reward[nei])

    # If the width of the confidence interval is < threshold, then break
    width = pes_safe_x.u - pes_safe_x.l
    if max(width[pes_safe_x.S_hat]) < args.thres_ci:
        print("\n Confidence interval is less than threshold STEP", i, "\n")
        break

    # Visualize using GP-Safety-Gym
    if args.render_gym:
        for p in path:
            pos_safety_gym = coord_to_safety_gym_pos(coord[p], world_shape)
            env.discreate_move(pos_safety_gym)

rec_point_exit_safe_exp = rec_point


# [STEP 2] Exploration and Exploitation of Reward

for i in range(args.max_iter_reward_opt):
    pes_safe_x.update_sets()
    p_safe = pes_safe_x.probability_safe()
    gp_reward.set_XY(np.vstack((gp_reward.X, coord[next_sample[0], :])),
                     np.vstack((gp_reward.Y, reward[next_sample[0]])))

    # reward w/ exploration bonus
    r_w_eb = reward_w_exp_bonus(pes_safe_x, reward_x)

    # Calculate the optimal policy
    next_sample, V, pi, P = calc_opt_policy(pes_safe_x, r_w_eb, cur_pos,
                                            args, bin_tran=False)

    # Update GP for safety and reward
    pes_safe_x.add_observation(*next_sample)
    reward_x.add_gp_observations(coord[next_sample[0], :],
                                 reward[next_sample[0]])

    # Reward history
    history_reward[rec_point] = reward[cur_pos]
    rec_point += 1

    # Visualize using GP-Safety-Gym
    if args.render_gym:
        pos_safety_gym = coord_to_safety_gym_pos(coord[cur_pos], world_shape)
        env.discreate_move(pos_safety_gym)

    # Move to the next position
    cur_pos = next_sample[0]

    if i % 50 == 0:
        print(i, "/", args.max_iter_reward_opt)
        print(cur_pos, int(V[cur_pos]), reward[cur_pos])

if args.render_gym:
    env.close()

file_name = "result/safe_near_opt_mdp_" + args.es2_type
np.savez(file_name, history_reward=history_reward)
