#!/usr/bin/env python

import re
import unittest
import numpy as np
import gym
import gym.spaces

from safety_gym.envs.engine import Engine


class TestBench(unittest.TestCase):
    def test_goal(self):
        ''' Point should run into and get a goal '''
        config = {
            'robot_base': 'xmls/point.xml',
            'goal_size': 0.5,
            'goal_placements': [(0, -.5, 5, .5)],
            'reward_goal': 1.0,
            'reward_distance': 1.0,
            'robot_locations': [(0, 0)],
            'robot_rot': 0,
            '_seed': 0,
        }
        env = Engine(config)
        env.reset()
        goal_met = False
        for _ in range(999):
            act = np.zeros(env.action_space.shape)
            act[0] = 1
            _, reward, done, info = env.step(act)
            self.assertFalse(done)
            # If we have not yet got the goal
            if not goal_met:
                # Reward should be positive, since we're moving towards it.
                self.assertGreater(reward, 0)
                # Update if we got the goal
                if 'goal_met' in info:
                    goal_met = info['goal_met']
                    # Assert we got 1 point for the goal
                    self.assertGreater(reward, 1)
            # env.render()  # Uncomment to visualize test
        self.assertTrue(goal_met)

    def test_hazards(self):
        ''' Point should run into and get a hazard '''
        config = {
            'robot_base': 'xmls/point.xml',
            'goal_size': 0.5,
            'goal_placements': [(5, -.5, 10, .5)],
            'reward_goal': 1.0,
            'reward_distance': 1.0,
            'constrain_indicator': True,
            'constrain_hazards': True,
            'hazards_num': 1,
            'hazards_size': 0.5,
            'hazards_locations': [(2, 0)],
            'hazards_cost': 1.0,
            'robot_locations': [(0, 0)],
            'robot_rot': 0,
            '_seed': 0,
        }
        env = Engine(config)
        env.reset()
        goal_met = False
        hazard_found = False
        for _ in range(999):
            act = np.zeros(env.action_space.shape)
            act[0] = 1
            _, reward, done, info = env.step(act)
            if not hazard_found:
                if info['cost']:
                    hazard_found = True
                    self.assertEqual(info['cost'], 1.0)  # Sparse costs
                    self.assertGreater(info['cost_hazards'], 0.0)  # Nonzero hazard cost
            if 'goal_met' in info:
                goal_met = info['goal_met']
            # env.render()  # Uncomment to visualize test
        self.assertTrue(hazard_found)
        self.assertTrue(goal_met)

    def test_vases(self):
        ''' Point should run into and past a vase, pushing it out of the way '''
        config = {
            'robot_base': 'xmls/point.xml',
            'goal_size': 0.5,
            'goal_placements': [(5, -.5, 10, .5)],
            'reward_goal': 1.0,
            'reward_distance': 1.0,
            'constrain_indicator': True,
            'constrain_vases': True,
            'vases_num': 1,
            'vases_locations': [(2, 0)],
            'vases_contact_cost': 1.0,
            'vases_displace_cost': 1.0,
            'vases_velocity_cost': 1.0,
            'robot_locations': [(0, 0)],
            'robot_rot': 0,
            '_seed': 0,
        }
        env = Engine(config)
        env.reset()
        goal_met = False
        vase_found = False
        for _ in range(999):
            act = np.zeros(env.action_space.shape)
            act[0] = 1
            _, reward, done, info = env.step(act)
            if not vase_found:
                if info['cost']:
                    vase_found = True
                    self.assertEqual(info['cost'], 1.0)  # Sparse costs
                    self.assertGreater(info['cost_vases_contact'], 0.0)  # Nonzero vase cost
                    self.assertGreater(info['cost_vases_velocity'], 0.0)  # Nonzero vase cost
            else:
                # We've already found the vase (and hit it), ensure displace cost
                self.assertEqual(info['cost'], 1.0)  # Sparse costs
                self.assertGreater(info['cost_vases_displace'], 0.0)  # Nonzero vase cost
            if 'goal_met' in info:
                goal_met = info['goal_met']
            # env.render()  # Uncomment to visualize test
        self.assertTrue(vase_found)
        self.assertTrue(goal_met)

    def check_correct_lidar(self, env_name):
        ''' Check that a benchmark env has the right lidar obs for the objects in scene '''
        env = gym.make(env_name)
        env.reset()
        physics = env.unwrapped
        world = physics.world
        obs_space_dict = physics.obs_space_dict
        task = physics.task
        lidar_count = sum('lidar' in o.lower() for o in obs_space_dict.keys())
        # Goal based lidar
        if task == 'x':
            self.assertEqual(lidar_count, 0)
        elif task == 'circle':
            self.assertEqual(lidar_count, 1)
            self.assertIn('circle_lidar', obs_space_dict)
        elif task == 'goal':
            self.assertIn('goal_lidar', obs_space_dict)
        elif task == 'push':
            self.assertIn('goal_lidar', obs_space_dict)
            self.assertIn('box_lidar', obs_space_dict)
        elif task == 'button':
            self.assertIn('goal_lidar', obs_space_dict)
            self.assertIn('buttons_lidar', obs_space_dict)

        if physics.constrain_hazards or physics.hazards_num > 0:
            self.assertIn('hazards_lidar', obs_space_dict)
            self.assertGreater(physics.hazards_num, 0)
        if physics.constrain_vases or physics.vases_num > 0:
            self.assertIn('vases_lidar', obs_space_dict)
            self.assertGreater(physics.vases_num, 0)
        if physics.constrain_pillars or physics.pillars_num > 0:
            self.assertIn('pillars_lidar', obs_space_dict)
            self.assertGreater(physics.pillars_num, 0)
        if physics.constrain_buttons or physics.buttons_num > 0:
            self.assertIn('buttons_lidar', obs_space_dict)
            self.assertGreater(physics.buttons_num, 0)
        if physics.constrain_gremlins or physics.gremlins_num > 0:
            self.assertIn('gremlins_lidar', obs_space_dict)
            self.assertGreater(physics.gremlins_num, 0)

    def test_correct_lidar(self):
        ''' We should have lidar for every object in the env '''
        matched = []
        for env_spec in gym.envs.registry.all():
            #if re.match(r'Safexp-.*-v0', env_spec.id) is not None:
            if 'Safexp' in env_spec.id and not('Vision' in env_spec.id):
                matched.append(env_spec.id)
        assert matched, 'Failed to match any environments!'
        for env_name in matched:
            print(env_name)
            self.check_correct_lidar(env_name)


if __name__ == '__main__':
    unittest.main()
