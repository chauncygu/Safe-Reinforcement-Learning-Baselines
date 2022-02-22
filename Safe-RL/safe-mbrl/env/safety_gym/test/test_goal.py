#!/usr/bin/env python

import unittest
import numpy as np

from safety_gym.envs.engine import Engine, ResamplingError


class TestGoal(unittest.TestCase):
    def rollout_env(self, env):
        ''' roll an environment until it is done '''
        done = False
        while not done:
            _, _, done, _ = env.step([1,0])

    def test_resample(self):
        ''' Episode should end with resampling failure '''
        config = {
            'robot_base': 'xmls/point.xml',
            'num_steps': 1001,
            'placements_extents': [-1, -1, 1, 1],
            'goal_size': 1.414,
            'goal_keepout': 1.414,
            'goal_locations': [(1, 1)],
            'robot_keepout': 1.414,
            'robot_locations': [(-1, -1)],
            'robot_rot': np.sin(np.pi / 4),
            'terminate_resample_failure': True,
            '_seed': 0,
        }
        env = Engine(config)
        env.reset()
        self.assertEqual(env.steps, 0)
        # Move the robot towards the goal
        self.rollout_env(env)
        # Check that the environment terminated early
        self.assertLess(env.steps, 1000)

        # Try again with the raise
        config['terminate_resample_failure'] = False
        env = Engine(config)
        env.reset()
        # Move the robot towards the goal, which should cause resampling failure
        with self.assertRaises(ResamplingError):
            self.rollout_env(env)


if __name__ == '__main__':
    unittest.main()
