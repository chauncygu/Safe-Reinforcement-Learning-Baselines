# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.wrappers.time_limit import TimeLimit
from envs.frozen_lake import FrozenLakeEnv
from envs.hopper import HopperEnv
from envs.cliff_envs import CliffCheetahEnv, CliffWalkerEnv
from envs.pusher import PusherEnv
from envs.peg_insertion import PegInsertionEnv

import numpy as np


def get_env(env_name, safety_param=0):
    # Reset reward should be in [-1, 0]
    if env_name in ['small-gridworld', 'large-gridworld']:
        if env_name == 'small-gridworld':
            map_name = '4x4'
            max_episode_steps = 30
            num_training_iterations = 20000
        else:
            map_name = '8x8'
            max_episode_steps = 100
            num_training_iterations = 20000
        env = FrozenLakeEnv(map_name=map_name)
        done_state = np.zeros(env.nS)
        done_state[0] = 1

        def reset_done_fn(s):
            return np.all(s == done_state)
        def reset_reward_fn(s, a):
            return float(reset_done_fn(s)) - 1.0

        agent_type = 'DDQNAgent'

    elif env_name == 'hopper':
        env = HopperEnv()

        def reset_reward_fn(s):
            height = s[0]
            ang = s[1]
            return (height > .7) and (abs(ang) < .2) - 1.0
        def reset_done_fn(s, a):
            return float(reset_done_fn(s)) - 1.0

        agent_type = 'DDPGAgent'
        max_episode_steps = 1000
        num_training_iterations = 1000000

    elif env_name == 'ball-in-cup':
        # Only import control suite if used. All other environments can be
        # used without the control suite dependency.
        from dm_control.suite.ball_in_cup import BallInCup
        env = BallInCup()
        reset_state = np.array([0., 0., 0., -0.05, 0., 0., 0., 0.])
        def reset_reward_fn(s):
            dist = np.linalg.norm(reset_state - s)
            return np.clip(1.0 - 0.5 * dist, 0, 1) - 1.0
        def reset_done_fn(s):
            return (reset_reward_fn(s) > 0.7)
        max_episode_steps = 50
        agent_type = 'DDPGAgent'
        num_training_iterations = 1000000

    elif env_name == 'peg-insertion':
        env = PegInsertionEnv()
        def reset_reward_fn(s, a):
            (forward_reward, reset_reward) = env.env._get_rewards(s, a)
            return reset_reward - 1.0
        def reset_done_fn(s):
            a = np.zeros(env.action_space.shape[0])
            return (reset_reward_fn(s, a) > 0.7)
        max_episode_steps = 50
        num_training_iterations = 1000000
        agent_type = 'DDPGAgent'

    elif env_name == 'pusher':
        env = PusherEnv()
        def reset_reward_fn(s, a):
            (forward_reward, reset_reward) = env.env._get_rewards(s, a)
            return reset_reward - 1.0
        def reset_done_fn(s):
            a = np.zeros(env.action_space.shape[0])
            return (reset_reward_fn(s, a) > 0.7)
        max_episode_steps = 100
        num_training_iterations = 1000000
        agent_type = 'DDPGAgent'

    elif env_name == 'cliff-walker':
        dist = 6
        env = CliffWalkerEnv()
        def reset_reward_fn(s, a):
            (forward_reward, reset_reward) = env.env._get_rewards(s, a)
            return (reset_reward > 0.7) - 1.0
        def reset_done_fn(s):
            a = np.zeros(env.action_space.shape[0])
            return (reset_reward_fn(s, a) > 0.7)
        max_episode_steps = 500
        num_training_iterations = 1000000
        agent_type = 'DDPGAgent'

    elif env_name == 'cliff-cheetah':
        dist = 14
        env = CliffCheetahEnv()
        def reset_reward_fn(s, a):
            (forward_reward, reset_reward) = env.env._get_rewards(s, a)
            return (reset_reward > 0.7) - 1.0
        def reset_done_fn(s):
            a = np.zeros(env.action_space.shape[0])
            return (reset_reward_fn(s, a) > 0.7)
        max_episode_steps = 500
        agent_type = 'DDPGAgent'
        num_training_iterations = 1000000

    else:
        raise ValueError('Unknown environment: %s' % env_name)

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    q_min = -1 * (1. - safety_param) * env._max_episode_steps
    lnt_params = {
        'reset_reward_fn': reset_reward_fn,
        'reset_done_fn': reset_done_fn,
        'q_min': q_min,
    }
    agent_params = {
        'num_training_iterations': num_training_iterations,
        'agent_type': agent_type,
    }
    return (env, lnt_params, agent_params)
