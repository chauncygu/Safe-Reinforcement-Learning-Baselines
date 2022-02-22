import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class LQR_Env(gym.Env):
 
    def __init__(self):

        self.viewer = None
 
        self.A =  np.array([[1.01, 0.01, 0.0],[0.01, 1.01, 0.01], [0., 0.01, 1.01]])
        self.B = np.eye(3)

        self.d, self.p = self.B.shape
        
        self.R = np.eye(self.p)
        self.Q = np.eye(self.d) / 1000
        
        self.time = 0

        self.action_space = spaces.Box(low=-1e+8, high=1e+8, shape=(self.p,))
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.d, ))

        self.state = np.random.normal(0,1,size = self.d)
        
        self._seed()

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):

        x = self.state

        cost = np.dot(x, np.dot(self.Q, x)) + np.dot(u, np.dot(self.R, u))
        new_x = np.dot(self.A, x) + np.dot(self.B, u) + self.np_random.normal(0,1,size = self.d)

        self.state = new_x

        terminated = False
        if self.time > 300:
            terminated = True

        self.time += 1
            
        return self._get_obs(), - cost, terminated, {}

    def _reset(self):
        self.state = self.np_random.normal(0, 1, size = self.d)
        self.last_u = None
        self.time = 0
        
        return self._get_obs()

    def _get_obs(self):
        return  self.state

    def get_params(self):
        return self.A, self.B, self.Q, self.R
