from __future__ import division, print_function, absolute_import

import GPy
import numpy as np
import mdptoolbox

import arguments

from safemdp.grid_world import compute_S_hat0

from utils.reward_utilities import RewardObj
from utils.safety_utilities import SafetyObj
from utils.mdp_utilities import (reward_w_exp_bonus, grid_world_graph_w_stay,
                                 calculate_P_grid, calculate_P_opti_pess)


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

# Safety threhsold, Lipschitz constant, scaling factors for confidence interval
h = args.h
L = args.L
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
pes_x = SafetyObj(gp_safety, world_shape, step_size, beta_safety,
                  safety, h, S0, S_hat0, L, update_dist=0)

# Define Optimistic Safety object
opt_x = SafetyObj(gp_safety, world_shape, step_size, beta_safety,
                  safety, h, S0, S_hat0, L, update_dist=0)

# Define Reward object
reward_x = RewardObj(gp_reward, beta_reward)

# Insert samples from (s, a) in S_hat0
tmp = np.arange(pes_x.coord.shape[0])
s_vec_ind = np.random.choice(tmp[np.any(pes_x.S_hat[:, 1:], axis=1)])
tmp = np.arange(1, pes_x.S.shape[1])
actions = tmp[pes_x.S_hat[s_vec_ind, 1:].squeeze()]
for i in range(3):
    pes_x.add_observation(s_vec_ind, np.random.choice(actions))
    opt_x.add_observation(s_vec_ind, np.random.choice(actions))

# Remove samples used for GP initialization
pes_x.gp.set_XY(pes_x.gp.X[n_samples:, :],
                pes_x.gp.Y[n_samples:])

opt_x.gp.set_XY(pes_x.gp.X[n_samples:, :],
                pes_x.gp.Y[n_samples:])

# Initial position
pos = int(npz_settings['start_pos'])

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

history_reward = np.zeros(args.max_time_steps)
history_pos = []

for i in range(args.max_iter_reward_opt):
    pes_x.update_sets()
    opt_x.update_opt_sets()
    opt_x.S_bar |= pes_x.S_hat

    G = grid_world_graph_w_stay(pes_x)
    P = calculate_P_grid(G, pes_x)
    P_pess, P_opti = calculate_P_opti_pess(P, pes_x, opt_x)

    # reward w/ exploration bonus
    r_w_eb = reward_w_exp_bonus(pes_x, reward_x)

    # Optimistic MDP
    pi_opti = mdptoolbox.mdp.PolicyIteration(P_opti, r_w_eb, args.gamma)
    pi_opti.run()
    V_opti = pi_opti.V

    # Pessimistic MDP
    pi_pess = mdptoolbox.mdp.PolicyIteration(P_pess, r_w_eb, args.gamma)
    pi_pess.run()
    V_pess = pi_pess.V

    # Value function for the interpolated MDP
    eta = 0.6
    V_ip = eta * np.array(V_opti) + (1 - eta) * np.array(V_pess)

    # Obtain the *safe* neighbors
    _set_safe_neighbor_nodes = [pos]
    _set_safe_actions = [0]
    for _, neigh_node, action in G.edges_iter(pos, data='action'):
        if action > 0:
            if pes_x.S_hat[pos, action]:
                _set_safe_neighbor_nodes.append(neigh_node)
                _set_safe_actions.append(action)

    max_V = -100
    for j in range(len(_set_safe_neighbor_nodes)):
        if V_ip[_set_safe_neighbor_nodes[j]] > max_V:
            next_pos = _set_safe_neighbor_nodes[j]
            act = _set_safe_actions[j]
            max_V = V_ip[next_pos]
        else:
            None

    # When no action except for "stay" is safe, next action is set to "stay"
    if max_V == -100:
        next_pos = pos
        act = 0

    history_pos.append(pos)

    # Update the current position
    next_sample = (next_pos, act)
    pos = next_pos

    # Update GPs for safety and reward
    pes_x.add_observation(*next_sample)
    opt_x.add_observation(*next_sample)
    reward_x.add_gp_observations(coord[pos, :], reward[pos])

    # Add GP observations for neighboring state
    if args.multi_obs:
        for _, nei, act in pes_x.graph.edges_iter(pos, data='action'):
            pes_x.add_observation(nei, act)
            opt_x.add_observation(nei, act)
            reward_x.add_gp_observations(coord[nei, :], reward[nei])

    # Reward history
    history_reward[i] = reward[pos]

    print(i, pos, reward[pos])

    # Visualize using GP-Safety-Gym
    if args.render_gym:
        pos_safety_gym = coord_to_safety_gym_pos(coord[pos], world_shape)
        env.discreate_move(pos_safety_gym)

if args.render_gym:
    env.close()

file_name = "result/safe_exp_opt"
np.savez(file_name, history_reward=history_reward)
