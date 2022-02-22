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

from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco import mujoco_env
import numpy as np
import os


def tolerance(x, bounds, margin):
    '''Returns 1 when x is within the bounds, and decays sigmoidally
    when x is within a certain margin outside the bounds.
    We've copied the function from [1] to reduce dependencies.

    [1] Tassa, Yuval, et al. "DeepMind Control Suite." arXiv preprint
    arXiv:1801.00690 (2018).
    '''
    (lower, upper) = bounds
    if lower <= x <= upper:
        return 0
    elif x < lower:
        dist_from_margin = lower - x
    else:
        assert x > upper
        dist_from_margin = x - upper
    loss_at_margin = 0.95
    w = np.arctanh(np.sqrt(loss_at_margin)) / margin
    s = np.tanh(w * dist_from_margin)
    return s*s


def huber(x, p):
    return np.sqrt(x*x + p*p) - p


class CliffCheetahEnv(HalfCheetahEnv):
    def __init__(self):
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_cheetah.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)

    def step(self, a):
        (s, _, done, info) = super(CliffCheetahEnv, self).step(a)
        r = self._get_rewards(s, a)[0]
        return (s, r, done, info)

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])

    def _get_rewards(self, s, a):
        (x, z, theta) = s[:3]
        xvel = s[9]
        # Reward the forward agent for running 9 - 11 m/s.
        forward_reward = (1.0 - tolerance(xvel, (9, 11), 7))
        theta_reward = 1.0 - tolerance(theta,
                                       bounds=(-0.05, 0.05),
                                       margin=0.1)
        # Reward the reset agent for being at the origin, plus
        # reward shaping to be near the origin and upright.
        reset_reward = 0.8 * (np.abs(x) < 0.5) + 0.1 * (1 - 0.2 * np.abs(x)) + 0.1 * theta_reward
        return (forward_reward, reset_reward)


class CliffWalkerEnv(Walker2dEnv):
    def __init__(self):
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/cliff_walker.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_filename, 5)

    def step(self, a):
        (s, _, done, info) = super(CliffWalkerEnv, self).step(a)
        r = self._get_rewards(s, a)[0]
        return (s, r, done, info)

    def _get_obs(self):
        '''Modified to include the x coordinate.'''
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[:], np.clip(qvel, -10, 10)]).ravel()

    def _get_rewards(self, s, a):
        x = s[0]
        running_vel = s[9] - 2.0
        torso_height = s[1]
        is_standing = float(torso_height > 1.2)
        is_falling = float(torso_height < 0.7)
        run_reward = np.clip(1 - 0.2 * huber(running_vel, p=0.1), 0, 1)
        stand_reward = np.clip(0.25 * torso_height +
                               0.25 * is_standing +
                               0.5 * (1 - is_falling), 0, 1)
        control_reward = np.clip(1 - 0.05 * np.dot(a, a), 0, 1)
        reset_location_reward = 0.8 * (np.abs(x) < 0.5) + 0.2 * (1 - 0.2 * np.abs(x))
        forward_reward = 0.5 * run_reward + 0.25 * stand_reward + 0.25 * control_reward
        reset_reward = 0.5 * reset_location_reward + 0.25 * stand_reward + 0.25 * control_reward
        return (forward_reward, reset_reward)


if __name__ == '__main__':
    import time
    # env = CliffCheetahEnv()
    env = CliffWalkerEnv()
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.01)
