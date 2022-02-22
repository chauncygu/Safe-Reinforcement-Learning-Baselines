'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:42:50
@LastEditTime: 2020-07-28 22:58:34
@Description:
'''

import numpy as np
import copy
from mbrl.optimizers import RandomOptimizer, CEMOptimizer


class MPC(object):
    optimizers = {"CEM": CEMOptimizer, "Random": RandomOptimizer}

    def __init__(self, mpc_config, reward_model = None):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"]
        conf = mpc_config[self.type]
        self.horizon = conf["horizon"]
        self.gamma = conf["gamma"]
        self.action_low = np.array(conf["action_low"]) # array (dim,)
        self.action_high = np.array(conf["action_high"]) # array (dim,)
        self.action_dim = conf["action_dim"]
        self.popsize = conf["popsize"]

        self.particle = conf["particle"]

        self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim])
            self.action_high = np.tile(self.action_high, [self.action_dim])
        
        self.optimizer = MPC.optimizers[self.type](sol_dim=self.horizon*self.action_dim,popsize=self.popsize,upper_bound=np.array(conf["action_high"]),lower_bound=np.array(conf["action_low"]),max_iters=conf["max_iters"],num_elites=conf["num_elites"],epsilon=conf["epsilon"],alpha=conf["alpha"])

        if reward_model is not None:
            self.reward_model = reward_model
            self.optimizer.setup(self.cost_function)
        else:
            self.optimizer.setup(self.cartpole_cost_function) # default cost function
        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])

    def act(self, model, state):
        '''
        :param state: model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.model = model
        self.state = state

        soln, var = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        if self.type == "CEM":
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        else:
            pass
        action = soln[0]
        return action

    def cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            state_next = self.model.predict(state, action) + state

            cost = -self.reward_model.predict(state_next, action)  # compute cost
            cost = cost.reshape(costs.shape)
            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def cartpole_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (batch_size x horizon number)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        # TODO: may be able to change to tensor like pets
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        actions = np.tile(actions, (self.particle, 1, 1))

        costs = np.zeros(self.popsize*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize*self.particle, axis=0)

        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (batch_size x action dim)

            state_next = self.model.predict(state, action) + state

            cost = self.cartpole_cost(state_next, action)  # compute cost
            costs += cost * self.gamma**t
            state = copy.deepcopy(state_next)

        # average between particles
        costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        return costs

    def cartpole_cost(self, state, action, env_cost=False, obs=True):
        """
        Calculate the cartpole env cost given the state

        Parameters:
        ----------
            @param numpy array - state : size should be (batch_size x state dim)
            @param numpy array - action : size should be (batch_size x action dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """

        if not obs:
            x = state[:, 0]
            x_dot = state[:, 1]
            theta = state[:, 2]
            theta_dot = state[:, 3]
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
        else:
            # self.add_bound = 0.8
            x = state[:, 0]
            x_dot = state[:, 1]
            cos_theta = state[:, 2]
            # todo: initially the GP may predict -1.xxx for cos
            # cos_theta[cos_theta < -1] = -1
            # cos_theta[cos_theta > 1] = 1
            sin_theta = state[:, 3]
            theta_dot = state[:, 4]

        action = action.squeeze()

        length = 0.5 # pole length
        x_tip_error = x - length*sin_theta
        y_tip_error = length - length*cos_theta
        reward = np.exp(-(x_tip_error**2 + y_tip_error**2)/length**2)

        self.action_cost = True
        self.x_dot_cost = True

        if self.action_cost:
            reward += -0.01 * action**2

        if self.x_dot_cost:
            reward += -0.001 * x_dot**2

        cost = -reward

        return cost
