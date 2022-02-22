'''PegInsertion environment.'''
import os
from gym.envs.mujoco import mujoco_env
import numpy as np


class PegInsertionEnv(mujoco_env.MujocoEnv):
    '''PegInsertionEnv.

    We define the forward task to be pulling the peg out of the hole, and the
    reset task to be putting the peg into the hole.
    '''

    def __init__(self, task='insert', sparse=False):
        self._sparse = sparse
        self._task = task
        envs_folder = os.path.dirname(os.path.abspath(__file__))
        xml_filename = os.path.join(envs_folder,
                                    'assets/peg_insertion.xml')
        super(PegInsertionEnv, self).__init__(xml_filename, 5)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        done = False

        (insert_reward, remove_reward) = self._get_rewards(obs, a)
        if self._task == 'insert':
            reward = insert_reward
        elif self._task == 'remove':
            reward = remove_reward
        else:
            raise ValueError('Unknown task: %s' % self._task)
        return obs, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        if self._task == 'insert':
            # Reset peg above hole:
            qpos = np.array([0.44542705, 0.64189252, -0.39544481, -2.32144865,
                             -0.17935136, -0.60320289, 1.57110214])
        else:
            # Reset peg in hole
            qpos = np.array([0.52601062,  0.57254126, -2.0747581, -1.55342248,
                             0.15375072, -0.5747922,  0.70163815])
        qvel = np.zeros(7)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_rewards(self, s, a):
        '''Compute the forward and reset rewards.
        Note: We assume that the reward is computed on-policy, so the given
        state is equal to the current observation.
        '''
        assert np.all(s == self._get_obs())
        peg_pos = np.hstack([self.get_body_com('leg_bottom'),
                            self.get_body_com('leg_top')])
        peg_bottom_z = peg_pos[2]
        goal_pos = np.array([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])
        start_pos = np.array([0.10600084, 0.15715909, 0.1496843, 0.24442536,
                              -0.09417238, 0.23726938])
        dist_to_goal = np.linalg.norm(goal_pos - peg_pos)
        dist_to_start = np.linalg.norm(start_pos - peg_pos)

        peg_to_goal_reward = np.clip(1. - dist_to_goal, 0, 1)
        peg_to_start_reward = np.clip(1. - dist_to_start, 0, 1)
        control_reward = np.clip(1 - 0.1 * np.dot(a, a), 0, 1)
        in_hole_reward = dist_to_goal < 0.1 and self.get_body_com("leg_bottom")[2] < -0.45

        if self._sparse:
            insert_reward = 0.8 * in_hole_reward + 0.2 * control_reward
        else:
            insert_reward = 0.5 * in_hole_reward + 0.25 * control_reward + 0.25 * peg_to_goal_reward
        remove_reward = 0.8 * peg_to_start_reward + 0.2 * control_reward
        return (insert_reward, remove_reward)

    def _get_obs(self):
        obs = np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
        ])
        return obs

    def camera_setup(self):
        pose = self.camera.get_pose()
        self.camera.set_pose(lookat=pose.lookat,
                             distance=pose.distance,
                             azimuth=270.,
                             elevation=-30.)


if __name__ == '__main__':
    import time
    env = PegInsertionEnv()
    env.reset()
    for _ in range(10000):
        action = env.action_space.sample()
        env.step(action)
        env.render()
        time.sleep(0.01)
