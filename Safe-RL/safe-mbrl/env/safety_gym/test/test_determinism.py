#!/usr/bin/env python

import unittest
import numpy as np
import gym
import safety_gym  # noqa


class TestDeterminism(unittest.TestCase):
    def check_qpos(self, env_name):
        ''' Check that a single environment is seed-stable at init '''
        for seed in [0, 1, 123456789]:
            print('running', env_name, seed)
            env1 = gym.make(env_name)
            env1.seed(np.random.randint(123456789))
            env1.reset()
            env1.seed(seed)
            env1.reset()
            env2 = gym.make(env_name)
            env2.seed(seed)
            env2.reset()
            np.testing.assert_almost_equal(env1.unwrapped.data.qpos, env2.unwrapped.data.qpos)

    def test_qpos(self):
        ''' Run all the bench envs '''
        for env_spec in gym.envs.registry.all():
            if 'Safexp' in env_spec.id:
                self.check_qpos(env_spec.id)

    def check_names(self, env_name):
        ''' Check that all the names in the mujoco model are the same for different envs '''
        print('check names', env_name)
        env1 = gym.make(env_name)
        env1.seed(0)
        env1.reset()
        env2 = gym.make(env_name)
        env2.seed(1)
        env2.reset()
        model1 = env1.unwrapped.model
        model2 = env2.unwrapped.model
        shared_names = ['actuator_names', 'body_names', 'camera_names', 'geom_names',
                        'joint_names', 'light_names', 'mesh_names', 'sensor_names',
                        'site_names', 'tendon_names', 'userdata_names']
        for n in shared_names:
            self.assertEqual(getattr(model1, n), getattr(model2, n))

    def test_names(self):
        ''' Run all the bench envs '''
        for env_spec in gym.envs.registry.all():
            if 'Safexp' in env_spec.id:
                self.check_names(env_spec.id)


if __name__ == '__main__':
    unittest.main()
