import gym
import safety_gym
import numpy as np

from safety_gym.envs.engine import Engine
from mujoco_py import (MjViewer, const, MjRenderContextOffscreen)


MAP_SIZE = 3.5

def reward_color(reward_value):
    """ Change the color depending on the reward function value
    High: yellow, medium: green, low: blue """
    rgb_alpha = np.zeros(4)
    rgb_alpha[0] = 2*(reward_value-0.5)
    rgb_alpha[1] = min(1, 2*reward_value)
    rgb_alpha[2] = 1 - reward_value
    rgb_alpha[3] = 1
    rgb_alpha *= 0.5
    return rgb_alpha


def coord_to_safety_gym_pos(coord, world_shape):
    _a = MAP_SIZE / world_shape[0]
    _pos_x = _a * (4*coord[0] - world_shape[0] + 1)
    _pos_y = _a * (4*coord[1] - world_shape[1] + 1)
    return np.array([_pos_x, _pos_y])


class Engine_GP(Engine):
    def __init__(self, config, reward_map, safety_map, world_shape):
        super(Engine_GP, self).__init__(config)
        self.reward_map = reward_map
        self.safety_map = safety_map
        self.world_shape = world_shape
        # 1/2 of the side of grid (i.e., square)
        self.a = MAP_SIZE / world_shape[0]

    def reward_for_render(self):
        reward = self.reward_map
        render_reward = reward.reshape(self.world_shape)/max(reward)
        return render_reward

    def safety_for_render(self, coef=0.8):
        safety = self.safety_map
        positive_safety = safety - min(safety)
        render_safety = coef * positive_safety.reshape(self.world_shape) / max(positive_safety)
        return render_safety
    
    def render_gp(self):
        """ Render the reward and safety functions in a single map
        Reward is represented by color (yellow: high, gree: medium, blue: low)
        Safety is represented by height """

        # First render the original part of Safety_Gym
        self.render()
        # Render a red ball directly above the true robot position
        virtual_pos = self.world.robot_pos().copy()
        virtual_pos[2] += 1
        self.render_sphere(virtual_pos, 0.07, np.array([1, 0, 0, 1]), alpha=1)
        self.viewer.add_marker(pos=(virtual_pos[0], virtual_pos[1], 0),
                               size=(0.02, 0.02, virtual_pos[2]),
                               rgba=np.array([1, 0, 0, 1]))

        # Reward and safety for rendering
        render_reward = self.reward_for_render()
        render_safety = self.safety_for_render()
        for i in range(self.world_shape[0]):
            for j in range(self.world_shape[1]):
                color = reward_color(render_reward[i, j])
                self.viewer.add_marker(pos=(self.a*(2*i - self.world_shape[0] + 1),
                                            self.a*(2*j - self.world_shape[1] + 1), 
                                            0),
                                       size=(self.a, self.a, render_safety[i, j]),
                                       rgba=np.array(color), label='')

    def discreate_move(self, pos):
        """ Move to the specified position """
        # Adjust the direction to the specified position
        while abs(self.ego_xy(pos)[1]) > 1e-2 or self.ego_xy(pos)[0] < 0:
            act = [0, 0.15]
            assert self.action_space.contains(act)
            obs, _, done, info = self.step(act)
            self.done = False
            assert self.observation_space.contains(obs)
            self.render_gp()

        # Adjust the distance to the specified position
        while abs(self.ego_xy(pos)[0]) > 5e-3:
            act = [0.005, 0]
            assert self.action_space.contains(act)
            obs, _, done, info = self.step(act)
            self.done = False
            assert self.observation_space.contains(obs)
            self.render_gp()
