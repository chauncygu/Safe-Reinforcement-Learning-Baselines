'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-24 10:42:50
@LastEditTime: 2020-07-29 21:34:37
@Description:
'''

import numpy as np
from mbrl.optimizers import RandomOptimizer, CEMOptimizer, RCEOptimizer

class SafeMPC(object):
    optimizers = {"CEM": CEMOptimizer, "RANDOM": RandomOptimizer, "RCE": RCEOptimizer}

    def __init__(self, env, mpc_config, cost_model = None, n_ensembles=0):
        # mpc_config = config["mpc_config"]
        self.type = mpc_config["optimizer"].upper()
        self.horizon = mpc_config["horizon"]
        self.gamma = mpc_config["gamma"]
        self.beta = 0.4
        self.n_ensembles = n_ensembles
        self.action_low = np.array(env.action_space.low) # array (dim,)
        self.action_high = np.array(env.action_space.high) # array (dim,)
        self.action_dim = env.action_space.shape[0]

        # self.popsize = conf["popsize"]
        # self.particle = conf["particle"]
        # self.init_mean = np.array([conf["init_mean"]] * self.horizon)
        # self.init_var = np.array([conf["init_var"]] * self.horizon)

        if len(self.action_low) == 1: # auto fill in other dims
            self.action_low = np.tile(self.action_low, [self.action_dim]) # (act dim)
            self.action_high = np.tile(self.action_high, [self.action_dim ])
       
        lb = np.tile(self.action_low, [self.horizon])
        ub = np.tile(self.action_high, [self.horizon])

        self.sol_dim = self.horizon*self.action_dim

        optimizer_config = mpc_config[self.type]
        self.popsize = optimizer_config["popsize"]
        
        self.optimizer = SafeMPC.optimizers[self.type](sol_dim=self.sol_dim,
            upper_bound=ub, lower_bound=lb, **optimizer_config)

        assert cost_model is not None, " cost model is not defined! "
        self.cost_model = cost_model

        if self.type == 'RCE':
            if self.n_ensembles>0:
                self.optimizer.setup(self.ts_cost_function)
            else:
                self.optimizer.setup(self.rce_cost_function)
        else:
            self.optimizer.setup(self.cost_function)

        self.reset()

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        #print('set init mean to 0')
        # self.prev_sol = np.tile((self.action_low + self.action_high) / 2, [self.horizon])
        # self.init_var = np.tile(np.square(self.action_low - self.action_high) / 16, [self.horizon])
        pass

    def act(self, model, state):
        '''
        :param state: model, (numpy array) current state
        :return: (float) optimal action
        '''
        self.model = model
        self.state = state

        soln, var = self.optimizer.obtain_solution()
        # if self.type == "CEM":
        #     self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
        # else:
        #     pass
        #print(soln)
        action = soln[:self.action_dim]
        return action

    def cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (popsize x sol_dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        #actions = np.tile(actions, (self.particle, 1, 1)) # 
        costs = np.zeros(self.popsize)#*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize, axis=0) # [pop size, state dim]
        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (pop size x action dim)
            x = np.concatenate((state, action), axis=1)
            state_next = self.model.predict(x) #+ state
            cost = self.cost_model.predict(state_next)  # compute cost
            cost = cost.reshape(costs.shape)
            costs += cost * self.gamma**t
            state = state_next
        # average between particles
        #costs = np.mean(costs.reshape((self.particle, -1)), axis=0)
        #if np.min(costs)>=200:
        #    print("all sampling traj will violate constraints")
        return costs

    def rce_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (popsize x sol_dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        #actions = np.tile(actions, (self.particle, 1, 1)) # 
        cost_rewards = np.zeros(self.popsize)#*self.particle)
        cost_constraints = np.zeros(self.popsize)#*self.particle)
        state = np.repeat(self.state.reshape(1, -1), self.popsize, axis=0) # [pop size, state dim]
        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (pop size x action dim)
            x = np.concatenate((state, action), axis=1)
            state_next = self.model.predict(x) #+ state

            cost_reward = self.cost_model._predict_reward(state_next)  # compute cost
            cost_reward = cost_reward.reshape(cost_rewards.shape)
            cost_rewards += cost_reward * self.gamma**t

            cost_const = self.cost_model._predict_cost(state_next)  # compute cost
            cost_const = cost_const.reshape(cost_constraints.shape)
            cost_constraints += cost_const * self.beta**t
            state = state_next

        return cost_rewards, cost_constraints

    def ts_cost_function(self, actions):
        """
        Calculate the cost given a sequence of actions
        Parameters:
        ----------
            @param numpy array - actions : size should be (popsize x sol_dim)

        Return:
        ----------
            @param numpy array - cost : length should be of batch_size
        """
        actions = actions.reshape((-1, self.horizon, self.action_dim)) # [pop size, horizon, action_dim]
        #actions = np.tile(actions, (self.particle, 1, 1)) # 
        cost_rewards = np.zeros((self.popsize, self.n_ensembles))
        cost_constraints = np.zeros((self.popsize, self.n_ensembles))
        state = np.repeat(self.state.reshape(1, -1), self.popsize, axis=0) # [pop size, state dim]
        state = np.tile(state.reshape((self.popsize, 1, -1)), [1, self.n_ensembles, 1] ) # [pop size, n_ensembles, state dim]
        for t in range(self.horizon):
            action = actions[:, t, :]  # numpy array (pop size x action dim)
            action = np.tile(action.reshape((self.popsize, 1, 2)), [1, self.n_ensembles, 1] ) # [pop size, n_ensembles, action dim]

            x = np.concatenate((state, action), axis=2)
            state_next = self.model.predict_with_each_model(x) # [pop size, n_ensembles, state dim]

            for i in range(self.n_ensembles):
                cost_reward = self.cost_model._predict_reward(state_next[:,i,:])  # [pop size, 1]
                cost_reward = np.squeeze(cost_reward)
                cost_rewards[:,i] += cost_reward * self.gamma**t

                cost_const = self.cost_model._predict_cost(state_next[:,i,:])  # compute cost
                cost_const = np.squeeze(cost_const)
                cost_constraints[:,i] += cost_const * self.beta**t
            state = state_next
        cost_r = np.mean(cost_rewards, axis=1)
        cost_c = np.max(cost_constraints, axis=1)
        return cost_r, cost_c