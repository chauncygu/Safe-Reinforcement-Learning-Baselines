"""
Model of linear environment to simulate the system and generate data
"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy.io
import random
from Abgabe.Normalize.MinMax import minmax_norm


class LinearEnv(gym.Env):
    """
        Class: Linear Environment
    """

    def __init__(self, path_dist, disturbance=0, nb_tracking=0, Q=np.eye(2), R=np.eye(2), reward_shaping=False):
        """
        Initialize linear Environment
         :param path_dist: path to external disturbances
        :param disturbance: 0-no disturbances, 1-external disturbance
        :param nb_tracking: future steps of tracking trajectory
        :param Q: weighting matrix of cost function
        :param R: weighting matrix of cost function
        """

        """----INIT PARAMETERS---------------------------------------------------------------------------------------"""
        # initial reference values
        self.E_ref_bat = 100000  
        self.T_ref = 21.5
        self.ref = np.array([[self.T_ref], [self.E_ref_bat]])
        self.steplength = 100  # how often the reference values change
        self.nb_maxtracking = nb_tracking  # plus one to always get the current value
        self.maxtracking = self.steplength * [self.T_ref]

        self.k = 0

        # initial cost values
        self.cost = 0
        self.gamma = 1e-8
        self.Q = Q
        self.R = R
        self.reward_shaping = reward_shaping

        # System Matrices
        self.A = np.array([[0.8511, 0],
                           [0, 0.99999]])

        self.B = np.array([[4, 0],
                           [0, -2500]])

        self.E = np.array([[0.022217, 0.0017912, 0.042212],
                           [0, 0, 0]])

        # init states
        self.x = np.array([[self.T_ref], [self.E_ref_bat]])
        # state constraints
        self.lbx = np.array([[20], [0]])
        self.ubx = np.array([[25], [200000]])
        # input constrains
        self.lbu = np.array([[-1], [-1]])
        self.ubu = np.array([[1], [1]])

        self.constraint_violations = 0  # counter for the number of constraint violations

        # load disturb. mat file
        self.dist = disturbance  # 0 -> no disturbance , 1 -> added disturbance
        dist = scipy.io.loadmat(path_dist)
        self.d = np.array([[0], [0], [0]])
        self.int_gains = dist['int_gains']
        self.room_temp = dist['room_temp']
        self.sol_rad = dist['sol_rad']
        self.room_temp_min = min(self.room_temp)
        self.room_temp_max = max(self.room_temp)
        self.int_gains_min = min(self.int_gains)
        self.int_gains_max = max(self.int_gains)
        self.sol_rad_min = min(self.sol_rad)
        self.sol_rad_max = max(self.sol_rad)
        # action and state space with bounds
        self.action_space = spaces.Box(low=self.lbu, high=self.ubu, dtype=np.float32)
        self.observation_space = spaces.Box(low=self.lbx, high=self.ubx, dtype=np.float32)

        # init plot variables
        self.u1_plot = []
        self.u2_plot = []
        self.T_plot = []
        self.Ebat_plot = []
        self.T_ref_plot = []
        self.steps = 0
        self.seed()

    def seed(self, seed=None):
        """
        Generate a random seed
        :param seed:
        :return:
        """
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Main function
        - Calculates new state
        - Calculates cost
        :param action: input to be applied to the system
        :return: new state, cost
        """

        """ calculate new state-------------------------------------------------------------------------------------"""
        # get action
        action = np.reshape(action, (2, 1))
        self.u1_plot = np.append(self.u1_plot, action[0])
        self.u2_plot = np.append(self.u2_plot, action[1])

        # get disturbance
        self.k = self.steps % self.room_temp.shape[0]
        if self.dist == 0:
            self.d = np.array([[0], [0], [0]])

        else:
            self.d = np.array([[self.room_temp.item(self.k)], [self.sol_rad.item(self.k)],
                               [self.int_gains.item(self.k)]])

        u = np.reshape(action, (2, 1))

        # calculate new state given action
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u) + np.dot(self.E, self.d)

        # update tracking
        reference = np.array([[self.maxtracking[0]], [self.E_ref_bat]])
        self.tracking_ref()

        """ calculate cost-------------------------------------------------------------------------------------------"""
        # state constraints for reward shaping
        cost_const = 0
        if self.reward_shaping == 'Rewardshaping':
            if self.x[0][0] > self.ubx[0][0]:
                cost_const += abs(self.x[0][0] - self.ubx[0][0]) * 5
                self.constraint_violations += 1
            if self.x[0][0] < self.lbx[0][0]:
                cost_const += abs(self.x[0][0] - self.lbx[0][0]) * 5
                self.constraint_violations += 1
            if self.x[1][0] > self.ubx[1][0]:
                cost_const += abs(self.x[1][0] - self.ubx[1][0]) * 0.00075
                self.constraint_violations += 1
            if self.x[1][0] < self.lbx[1][0]:
                cost_const += abs(self.x[1][0] - self.lbx[1][0]) * 0.00075
                self.constraint_violations += 1
        # quadratic cost
        cost = 0.5 * np.dot((reference - self.x).T, np.dot(self.Q, (reference - self.x))) +\
               0.5 * np.dot(u.T, np.dot(self.R, u))

        # add costs
        self.cost = cost[0][0] + cost_const

        self.steps += 1

        return self._get_obs(), -self.cost

    def reset_states(self):
        """
        Function to reset states
        :return: new states
        """
        # reset plot values
        self.u1_plot = []
        self.u2_plot = []
        self.T_plot = []
        self.Ebat_plot = []
        self.T_ref_plot = []

        self.k = 0

        # set state x0
        t = random.uniform(self.lbx[0][0], self.ubx[0][0])
        e = random.uniform(self.lbx[1][0], self.ubx[1][0])
        self.ref = np.array([[self.T_ref], [self.E_ref_bat]])
        self.x = np.array([[t], [e]])
        return self._get_obs()

    def _get_obs(self):
        """
        Function to normalize observation and save plot values
        :return: normalized states
        """

        self.T_plot = np.append(self.T_plot, self.x[0][0])
        self.Ebat_plot = np.append(self.Ebat_plot, self.x[1][0])
        self.T_ref_plot = np.append(self.T_ref_plot, self.maxtracking[0])

        # normalize state
        t = minmax_norm(self.x[0][0], self.lbx[0][0], self.ubx[0][0])
        e = minmax_norm(self.x[1][0], self.lbx[1][0], self.ubx[1][0])

        return np.array([[t], [e]])

    def get_val(self):
        """
        Function to return plot values
        :return: plot values
        """
        return [self.T_plot, self.Ebat_plot, self.u1_plot, self.u2_plot, self.T_ref_plot]

    def get_future_dist(self, nb_disturbance):
        """
        Get nb_disturbance future disturbance values
        :param nb_disturbance:
        :return:
        """
        dist = []
        test = int(nb_disturbance / 3)
        for i in range(test):
            count = (self.k + i) % self.room_temp.shape[0]
            dist.append(minmax_norm(self.room_temp.item(count), self.room_temp_min, self.room_temp_max))
            dist.append(minmax_norm(self.sol_rad.item(count), self.sol_rad_min, self.sol_rad_max))
            dist.append(minmax_norm(self.int_gains.item(count), self.int_gains_min, self.int_gains_max))
        return dist

    def get_future_tracking(self):
        """
        Function to return the future reference trajectory of the temperature
        :return:
        """
        ref = [0] * self.nb_maxtracking
        for i in range(self.nb_maxtracking):
            ref[i] = minmax_norm(self.maxtracking[i], self.lbx[0][0], self.ubx[0][0])
        return ref

    def tracking_ref(self):
        """
        Generate a tracking reference signal
        :return:
        """

        if self.steps % self.steplength == 0 and self.steps > self.nb_maxtracking:
            self.ref[0][0] = random.randrange(self.lbx[0]*10,  self.ubx[0]*10)/10
            self.ref[1][0] = self.E_ref_bat
            self.steplength = 25  # random.randrange(25, 100) -> possible to change the steplength

        self.maxtracking.pop(0)
        self.maxtracking.append(self.ref[0][0])
        return
