from envs.hovercraft import Hovercraft
from nets.policy_net import PolicyNetwork
from nets.vertex_policy_net import VertexPolicyNetwork
from nets.value_net import ValueNetwork
from utils.replay_buffer import ReplayBuffer
from algos.ddpy import DDPG

import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pickle
import os
import sys

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

lamb = 1
seed = 10
safe_th = 0.25

env = Hovercraft(fixed_init=True, fixed_target=True, safe_th=safe_th, lamb=lamb)
obs_dim = env.obs_dim
action_dim = env.action_dim
num_vertex = env.num_vertex
hidden_dim = 256
num_episodes = 300
num_steps = 100  # 5 seconds
batch_size = 128
value_lr = 1e-3
policy_lr = 1e-4

parent_dir = os.getcwd()


def train_agent(path,
                env,
                agent,
                seed=0,
                num_episodes=100,
                num_steps=100,
                batch_size=128,
                replay_buffer_size=1000000):

    if not os.path.isdir(path):
        os.makedirs(path)
    os.chdir(path)

    env.seed(seed)
    random.seed(seed)

    pickle.dump(agent.policy_net, open('first_policy.pickle', 'wb'))

    replay_buffer = ReplayBuffer(replay_buffer_size)

    rewards = []
    max_th_list = []
    ave_th_list = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        max_th = 0
        ave_th = 0
        for step in range(num_steps):
            action = agent.policy_net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > batch_size:
                agent.train_step(replay_buffer=replay_buffer, batch_size=batch_size)

            state = next_state
            episode_reward += reward
            max_th = max(max_th, abs(state[2]))
            ave_th += abs(state[2])

        rewards.append(episode_reward)
        max_th_list.append(max_th)
        ave_th_list.append(ave_th / num_steps)

    pickle.dump(agent.policy_net, open('last_policy.pickle', 'wb'))
    pickle.dump(rewards, open('rewards.pickle', 'wb'))
    pickle.dump(max_th_list, open('max_angle.pickle', 'wb'))
    pickle.dump(ave_th_list, open('ave_angle.pickle', 'wb'))

    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Reward vs Episode')
    plt.savefig('rewards.png', dpi=100)
    plt.close()


"""
Policy Net
"""

torch.manual_seed(seed)

path = os.path.join(parent_dir, 'results/hovercraft_' + str(safe_th) + '_' + str(seed) + '/pn')

value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
target_policy_net = PolicyNetwork(env=env, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

agent = DDPG(policy_net=policy_net, value_net=value_net,
             target_policy_net=target_policy_net, target_value_net=target_value_net,
             value_lr=value_lr, policy_lr=policy_lr)

train_agent(path=path, env=env, agent=agent, seed=seed, num_episodes=num_episodes, num_steps=num_steps)


"""
Simple Vertex Policy Net
"""

env.dummy_vertex = True

torch.manual_seed(seed)

path = os.path.join(parent_dir, 'results/hovercraft_' + str(safe_th) + '_' + str(seed) + '/svpn')

value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
policy_net = VertexPolicyNetwork(env=env, obs_dim=obs_dim, num_vertex=num_vertex, hidden_dim=hidden_dim).to(device)

target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
target_policy_net = VertexPolicyNetwork(env=env, obs_dim=obs_dim, num_vertex=num_vertex, hidden_dim=hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

agent = DDPG(policy_net=policy_net, value_net=value_net,
             target_policy_net=target_policy_net, target_value_net=target_value_net,
             value_lr=value_lr, policy_lr=policy_lr)

train_agent(path=path, env=env, agent=agent, seed=seed, num_episodes=num_episodes, num_steps=num_steps)


"""
Vertex Policy Net
"""

env.dummy_vertex = False

torch.manual_seed(seed)

path = os.path.join(parent_dir, 'results/hovercraft_' + str(safe_th) + '_' + str(seed) + '/vpn')

value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
policy_net = VertexPolicyNetwork(env=env, obs_dim=obs_dim, num_vertex=num_vertex, hidden_dim=hidden_dim).to(device)

target_value_net  = ValueNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
target_policy_net = VertexPolicyNetwork(env=env, obs_dim=obs_dim, num_vertex=num_vertex, hidden_dim=hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

agent = DDPG(policy_net=policy_net, value_net=value_net,
             target_policy_net=target_policy_net, target_value_net=target_value_net,
             value_lr=value_lr, policy_lr=policy_lr)

train_agent(path=path, env=env, agent=agent, seed=seed, num_episodes=num_episodes, num_steps=num_steps)


