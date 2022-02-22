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


from coach_util import Transition, RunPhase
from gym import Wrapper
from gym.wrappers.time_limit import TimeLimit
import numpy as np


class SafetyWrapper(Wrapper):
    # TODO: allow user to specify number of reset attempts. Currently fixed at 1.
    def __init__(self, env, reset_agent, reset_reward_fn, reset_done_fn, q_min):
        '''
        A SafetyWrapper protects the inner environment from danerous actions.

        args:
            env: Environment implementing the Gym API
            reset_agent: an agent implementing the coach Agent API
            reset_reward_fn: a function that returns the reset agent reward
                for a given observation.
            reset_done_fn: a function that returns whether the reset agent
                has successfully reset.
            q_min: a float that is used to decide when to do early aborts.
        '''
        assert isinstance(env, TimeLimit)
        super(SafetyWrapper, self).__init__(env)
        self._reset_agent = reset_agent
        self._reset_agent.exploration_policy.change_phase(RunPhase.TRAIN)
        self.env._reset_reward_fn = reset_reward_fn
        self.env._reset_done_fn = reset_done_fn
        self._max_episode_steps = env._max_episode_steps
        self._q_min = q_min
        self._obs = env.reset()

        # Setup internal structures for logging metrics.
        self._total_resets = 0  # Total resets taken during training
        self._episode_rewards = []  # Rewards for the current episode
        self._reset_history = []
        self._reward_history = []

    def _reset(self):
        '''Internal implementation of reset() that returns additional info.'''
        obs = self._obs
        obs_vec = [np.argmax(obs)]
        for t in range(self._max_episode_steps):
            (reset_action, _) = self._reset_agent.choose_action(
                {'observation': obs[:, None]}, phase=RunPhase.TRAIN)
            (next_obs, r, _, info) = self.env.step(reset_action)
            reset_reward = self.env._reset_reward_fn(next_obs, reset_action)
            reset_done = self.env._reset_done_fn(next_obs)
            transition = Transition({'observation': obs[:, None]},
                                    reset_action, reset_reward,
                                    {'observation': next_obs[:, None]},
                                    reset_done)
            self._reset_agent.memory.store(transition)
            obs = next_obs
            obs_vec.append(np.argmax(obs))
            memory_size = self._reset_agent.memory.num_transitions_in_complete_episodes()
            if memory_size > self._reset_agent.tp.batch_size:
                # Do one training iteration of the reset agent
                self._reset_agent.train()
            if reset_done:
                break
        if not reset_done:
            obs = self.env.reset()
            self._total_resets += 1

        # Log metrics
        self._reset_history.append(self._total_resets)
        self._reward_history.append(np.mean(self._episode_rewards))
        self._episode_rewards = []

        # If the agent takes an action that causes an early abort the agent
        # shouldn't believe that the episode terminates. Because the reward is
        # negative, the agent would be incentivized to do early aborts as
        # quickly as possible. Thus, we set done = False.
        done = False

        # Reset the elapsed steps back to 0
        self.env._elapsed_steps = 0
        return (obs, r, done, info)

    def reset(self):
        (obs, r, done, info) = self._reset()
        return obs

    def step(self, action):
        reset_q = self._reset_agent.get_q(self._obs, action)
        if reset_q < self._q_min:
            (obs, r, done, info) = self._reset()
        else:
            (obs, r, done, info) = self.env.step(action)
            self._episode_rewards.append(r)
        self._obs = obs
        return (obs, r, done, info)

    def plot_metrics(self, output_dir='/tmp'):
        '''
        Plot metrics collected during training.

        args:
            output_dir: (optional) folder path for saving results.
        '''

        import matplotlib.pyplot as plt
        import json
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data = {
            'reward_history': self._reward_history,
            'reset_history': self._reset_history
        }
        with open(os.path.join(output_dir, 'data.json'), 'w') as f:
            json.dump(data, f)

        # Prepare data for plotting
        rewards = np.array(self._reward_history)
        lnt_resets = np.array(self._reset_history)
        num_episodes = len(rewards)
        baseline_resets = np.arange(num_episodes)
        episodes = np.arange(num_episodes)

        # Plot the data
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.gca()
        ax2 = ax1.twinx()
        ax1.plot(episodes, rewards, 'g.')
        ax2.plot(episodes, lnt_resets, 'b-')
        ax2.plot(episodes, baseline_resets, 'b--')

        # Label the plot
        ax1.set_ylabel('average step reward', color='g', fontsize=20)
        ax1.tick_params('y', colors='g')
        ax2.set_ylabel('num. resets', color='b', fontsize=20)
        ax2.tick_params('y', colors='b')
        ax1.set_xlabel('num. episodes', fontsize=20)
        plt.savefig(os.path.join(output_dir, 'plot.png'))

        plt.show()
