#!/usr/bin/env python

import unittest
import numpy as np
import joblib
import os
import os.path as osp
import gym
import safety_gym
from safety_gym.envs.engine import Engine


class TestObs(unittest.TestCase):
    def test_rotate(self):
        ''' Point should observe compass/lidar differently for different rotations '''
        config = {
            'robot_base': 'xmls/point.xml',
            'observation_flatten': False,
            'observe_sensors': False,
            'observe_remaining': False,
            'observe_goal_lidar': True,
            'observe_goal_comp': True, 
            'goal_size': 3,
            'goal_locations': [(5, 0)],
            'robot_locations': [(1, 1)],
            '_seed': 0,
        }
        for s in (2, 3):
            config['compass_shape'] = s
            config['robot_rot'] = 5.3
            env = Engine(config)
            obs0 = env.reset()
            # for _ in range(1000): env.render()
            # print('obs0', obs0)
            config['robot_rot'] = np.pi / 4
            env = Engine(config)
            obs1 = env.reset()
            # for _ in range(1000): env.render()
            # print('obs1', obs1)
            self.assertTrue((obs0['goal_lidar'] != obs1['goal_lidar']).any())
            self.assertTrue((obs0['goal_compass'] != obs1['goal_compass']).any())

    def test_spaces(self):
        ''' Observation spaces should not unintentionally change from known good reference '''
        BASE_DIR = os.path.dirname(safety_gym.__file__)
        fpath = osp.join(BASE_DIR, 'test', 'obs_space_refs.pkl')
        obs_space_refs = joblib.load(fpath)
        for env_spec in gym.envs.registry.all():
            if 'Safexp' in env_spec.id and env_spec.id in obs_space_refs:
                print('Checking obs space for... ', env_spec.id)
                env = gym.make(env_spec.id)
                ref_obs_space_dict = obs_space_refs[env_spec.id]
                obs_spaces_are_same = env.obs_space_dict==ref_obs_space_dict
                if not(obs_spaces_are_same):
                    print('\n', env_spec.id, '\n')
                    print('Current Observation Space:\n', env.obs_space_dict, '\n\n')
                    print('Reference Observation Space:\n', ref_obs_space_dict, '\n\n')
                self.assertTrue(obs_spaces_are_same)


if __name__ == '__main__':
    unittest.main()
