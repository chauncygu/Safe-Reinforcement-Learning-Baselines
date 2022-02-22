'''Pusher environment.'''
import os
from gym.envs.mujoco import mujoco_env
import numpy as np


class PusherEnv(mujoco_env.MujocoEnv):
    '''PusherEnv.'''
    GOAL_ZERO_POS = [0.45, -0.05, -0.3230]  # from xml
    OBJ_ZERO_POS = [0.45, -0.05, -0.275]  # from xml

    def __init__(self, task='forward'):

        self._task = task

        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/pusher.xml')
        super(PusherEnv, self).__init__(xml_filename, 5)

        # Note: self._goal is the same for the forward and reset tasks. Only
        # the reward function changes.
        (self._goal, self._start) = self._get_goal_start()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def _get_goal_start(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[:] = 0
        qvel[:] = 0
        self.set_state(qpos, qvel)
        goal = self.get_body_com('goal').copy()
        start = self.get_body_com('object').copy()
        return (goal, start)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        done = False

        (forward_shaped_reward, reset_shaped_reward) = self._get_rewards(obs, a)
        if self._task == 'forward':
            r = forward_shaped_reward
        elif self._task == 'reset':
            r = reset_shaped_reward
        else:
            raise ValueError('Unknown task: %s', self._task)
        return (obs, r, done, {})

    def _get_obs(self):
        obs = np.concatenate([self.model.data.qpos.flat[:5],
                              self.model.data.qvel.flat[:3],
                              self.get_body_com('tips_arm')[:2]])
        return obs

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:] = 0
        qpos[-4:-2] += self.np_random.uniform(-0.05, 0.05, 2)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[-4:] = 0

        # For the reset task, flip the initial positions of the goal and puck
        if self._task == 'reset':
            qpos[-3] -= 0.7
            qpos[-1] += 0.7

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _huber(self, x, bound, delta=0.2):
        assert delta < bound
        if x < delta:
            loss = 0.5 * x * x
        else:
            loss = delta * (x - 0.5 * delta)
        return loss

    def _reward_fn(self, x, bound=0.7):
        # Using bound = 0.7 because that's the initial puck-goal distance.
        x = np.clip(x, 0, bound)
        loss = self._huber(x, bound)
        loss /= self._huber(bound, bound)
        reward = 1 - loss
        assert 0 <= loss <= 1
        return reward

    def _get_rewards(self, s, a):
        del s
        if not hasattr(self, '_goal'):
            print('Warning: goal or start has not been set')
            return (0, 0)
        obj_to_arm = self.get_body_com('object') - self.get_body_com('tips_arm')
        obj_to_goal = self.get_body_com('object') - self._goal
        obj_to_start = self.get_body_com('object') - self._start
        obj_to_arm_dist = np.linalg.norm(obj_to_arm)
        obj_to_goal_dist = np.linalg.norm(obj_to_goal)
        obj_to_start_dist = np.linalg.norm(obj_to_start)
        control_dist = np.linalg.norm(a)

        forward_reward = self._reward_fn(obj_to_goal_dist)
        reset_reward = self._reward_fn(obj_to_start_dist)
        obj_to_arm_reward = self._reward_fn(obj_to_arm_dist)
        # The control_dist is between 0 and sqrt(2^2 + 2^2 + 2^2) = 3.464
        control_reward = self._reward_fn(control_dist, bound=3.464)

        forward_reward_vec = [forward_reward, obj_to_arm_reward, control_reward]
        reset_reward_vec = [reset_reward, obj_to_arm_reward, control_reward]

        reward_coefs = (0.5, 0.375, 0.125)
        forward_shaped_reward = sum(
            [coef * r for (coef, r) in zip(reward_coefs, forward_reward_vec)])
        reset_shaped_reward = sum(
            [coef * r for (coef, r) in zip(reward_coefs, reset_reward_vec)])

        assert 0 <= forward_shaped_reward <= 1
        assert 0 <= reset_shaped_reward <= 1

        return (forward_shaped_reward, reset_shaped_reward)


if __name__ == '__main__':
    import time
    env = PusherEnv(task='reset')
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.01)
