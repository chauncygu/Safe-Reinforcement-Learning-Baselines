#!/usr/bin/env python

import unittest
import numpy as np

from safety_gym.envs.engine import Engine, ResamplingError


class TestButton(unittest.TestCase):
    def rollout_env(self, env, gets_goal=False):
        '''
        Roll an environment out to the end, return final info dict.
        If gets_goal=True, then also assert that we got a goal successfully.
        '''
        got_goal = False
        done = False
        while not done:
            _, _, done, info = env.step([1, 0])
            if 'goal_met' in info:
                got_goal = True
        if gets_goal:
            self.assertTrue(got_goal)
        return info

    def test_timer(self):
        ''' Buttons should wait a period before becoming active again '''
        config = {
            'robot_base': 'xmls/point.xml',
            'num_steps': 100,
            'task': 'button',
            'buttons_num': 2,
            'buttons_locations': [(0, 0), (1, 0)],
            'buttons_resampling_delay': 1000,
            'constrain_buttons': True,
            'constrain_indicator': True,
            'robot_locations': [(-1, 0)],
            'robot_rot': 0,
            '_seed': 0,
        }
        # Correct button is pressed, nothing afterwards
        env = Engine(config)
        env.reset()
        info = self.rollout_env(env, gets_goal=True)
        self.assertEqual(info['cost_buttons'], 0.0)
        # Correct button is pressed, then times out and penalties
        config['buttons_resampling_delay'] = 10
        env = Engine(config)
        env.reset()
        info = self.rollout_env(env, gets_goal=True)
        self.assertEqual(info['cost_buttons'], 1.0)
        # Wrong button is pressed, gets penalty
        config['_seed'] = 1
        env = Engine(config)
        env.reset()
        info = self.rollout_env(env)
        self.assertEqual(info['cost_buttons'], 1.0)


if __name__ == '__main__':
    unittest.main()
