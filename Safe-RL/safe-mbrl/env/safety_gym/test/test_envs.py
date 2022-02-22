#!/usr/bin/env python

import unittest
import gym
import safety_gym.envs  # noqa


class TestEnvs(unittest.TestCase):
    def check_env(self, env_name):
        ''' Run a single environment for a single episode '''
        print('running', env_name)
        env = gym.make(env_name)
        env.reset()
        done = False
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

    def test_envs(self):
        ''' Run all the bench envs '''
        for env_spec in gym.envs.registry.all():
            if 'Safexp' in env_spec.id:
                self.check_env(env_spec.id)



if __name__ == '__main__':
    unittest.main()
