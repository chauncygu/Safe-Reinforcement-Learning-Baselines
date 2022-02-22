import time
import numpy as np
import cvxpy as cv

from gym_factored.envs.base import DiscreteEnv
from util.mdp import get_mdp_functions
from util.grb import *


class LinearProgrammingPlanner:
    def __init__(self,
                 transition: np.array,
                 reward: np.array,
                 cost: np.array,
                 terminal: np.array,
                 isd: np.array,
                 env,
                 max_reward,
                 min_reward,
                 max_cost,
                 min_cost,
                 horizon=3,
                 cost_bound=None,
                 verbose=False,
                 solver='cvxpy'):
        self.transition = transition
        self.isd = isd  # initial states distribution
        self.terminal = terminal
        self.reward = reward
        if cost is not None:
            self.cost = cost
        else:
            self.cost = np.zeros(shape=self.reward.shape)
        self.cost_bound = cost_bound
        self.env = env
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.max_cost = max_cost
        self.min_cost = min_cost

        self.ns, self.na = reward.shape
        self.horizon = horizon
        self.states = np.arange(self.ns)
        self.terminal_states = np.arange(self.ns)[terminal]
        self.non_terminal_states = np.arange(self.ns)[~terminal]
        self.actions = range(self.na)
        self.rng = np.random.default_rng()
        self.verbose = verbose
        if solver == 'grb' and GUROBI_FOUND:
            self.solver = 'grb'
        else:
            self.solver = 'cvxpy'

        self.policy = np.zeros(shape=(self.horizon, self.ns, self.na))

        self.lp = None
        self.x = None
        self.exp_cost = None
        self.exp_reward = None


        self.time_step = 0

    def act(self, state):
        action = self.rng.choice(self.actions, size=1, p=self.policy[self.time_step][state])[0]
        self.time_step += 1
        return action

    def add_transition(self, state, reward, action, next_state, done, info=None):
        pass

    @classmethod
    def from_discrete_env(cls, env: DiscreteEnv, **kwargs) -> 'LinearProgrammingPlanner':
        """
        gets as input a gym discrete env and returns a value iteration agent to solve
        :param env: a gym discrete env
        :param kwargs: args for planner
        :return: a planner
        """
        transition, reward, cost, terminal = get_mdp_functions(env)
        for s in np.arange(env.nS)[terminal]:
            transition[s, :, :] = 0
            transition[s, :, s] = 1
            reward[s, :] = 0
            cost[s, :] = 0
        max_reward, min_reward = reward.max(), reward.min()
        max_cost, min_cost = cost.max(), cost.min()
        return cls(transition, reward, cost, terminal, env.isd, env, max_reward, min_reward, max_cost, min_cost, **kwargs)

    def solve(self):
        t0 = time.perf_counter()
        if self.verbose:
            print("instantiating LP")
        self.instantiate_lp()
        if self.verbose:
            print("LP instantiated in {} seconds".format(time.perf_counter() - t0))
            t0 = time.perf_counter()

        self.solve_lp()
        if self.verbose:
            print("LP solved in {} seconds".format(time.perf_counter() - t0))
        if self.lp.status == cv.INFEASIBLE:
            print("LP is infeasible")
        if self.lp.status == cv.UNBOUNDED:
            print("LP is unbounded")
        # elif self.lp.status in [cv.OPTIMAL, cv.OPTIMAL_INACCURATE]:
        self.extract_policy()

    def solve_lp(self):
        if self.solver == 'grb':
            solve_gurobi_lp(self.lp, self.verbose)
            return self.lp.status == 2  # optimal
        else:
            self.lp.solve(verbose=self.verbose)
            return self.lp.status == cv.OPTIMAL

    def instantiate_lp(self):
        if self.solver == 'grb':
            self.instantiate_lp_grb()
        else:
            self.instantiate_lp_cvxpy()

    def instantiate_lp_cvxpy(self):
        # variables
        self.x = [
            cv.Variable(shape=(self.ns, self.na),
                        nonneg=True,
                        name="x[{}]".format(h))
            for h in range(self.horizon)
        ]

        # expressions
        self.exp_reward = cv.Constant(0)
        self.exp_cost = cv.Constant(0)
        for h in range(self.horizon):
            self.exp_reward += cv.sum(cv.multiply(self.x[h], self.reward))
            self.exp_cost += cv.sum(cv.multiply(self.x[h], self.cost))

        # objective
        obj = cv.Maximize(self.exp_reward)

        # constraints
        if self.horizon > 0:
            constraints = [
                # first time step outflow
                cv.sum(self.x[0][s]) ==
                # first time step inflow
                self.isd[s]
                for s in self.states
            ]
            for h in range(1, self.horizon):
                constraints += [
                    # outflow == inflow
                    cv.sum(self.x[h][s]) == cv.sum(cv.multiply(self.x[h - 1], self.transition[:, :, s]))
                    for s in self.non_terminal_states
                ] + [
                   cv.sum(self.x[h][self.terminal_states], axis=1) == 0
                ]
        else:
            constraints = []
        if self.cost_bound is not None:
            constraints.append(self.exp_cost <= self.cost_bound)

        # problem
        self.lp = cv.Problem(obj, constraints)

    def instantiate_lp_grb(self):
        self.lp = Model('ConstrainedMDP')
        if not self.verbose:
            self.lp.Params.OutputFlag = 0

        if self.verbose:
            print("adding variables")
        self.x = x = self.lp.addVars(range(self.horizon), self.states, self.actions, lb=0.0, name='x')

        if self.verbose:
            print("adding constraints")

        self.exp_reward = quicksum(
            quicksum(
                quicksum(
                    x[h, s, a] * self.reward[s, a]
                    for h in range(self.horizon)
                )
                for a in self.actions
            )
            for s in self.non_terminal_states
        )
        self.exp_cost = quicksum(
            quicksum(
                quicksum(
                    x[h, s, a] * self.cost[s, a]
                    for h in range(self.horizon)
                )
                for a in self.actions
            )
            for s in self.non_terminal_states
        )
        if self.cost_bound is not None:
            self.lp.addConstr(
                self.exp_cost <= self.cost_bound,
                'expected_cost_bound'
            )

        if self.horizon > 0:
            self.lp.addConstrs((
                quicksum(x[0, s, a] for a in self.actions) == self.isd[s]
                for s in self.states
            ), 'isd')

        self.lp.addConstrs((
            # out(suc)- in(suc) == isd(suc)
            quicksum(x[h, suc, a] for a in self.actions)
            == quicksum(
                quicksum(
                    self.transition[s, a, suc] * x[h-1, s, a]
                    for s in np.arange(self.ns)[self.transition[:, a, suc].nonzero()]
                    # for s in self.states
                ) for a in self.actions)
            for suc in self.non_terminal_states for h in range(1, self.horizon)
        ), 'transient')

        self.lp.addConstrs((
            quicksum(
                x[h, suc, a]
                for a in self.actions
            ) == 0
            for suc in self.states if self.terminal[suc] for h in range(self.horizon)
        ), 'terminal_flow')

        if self.verbose:
            print("setting objective")
        self.lp.setObjective(
            self.exp_reward,
            GRB.MAXIMIZE
        )

        self.lp.update()

    def get_policy(self, time_step=0):
        return self.policy[time_step]

    def extract_policy(self):
        return self.extract_ground_policy()

    def extract_ground_policy(self):
        for h in range(self.horizon):
            for s in self.states:
                self.set_policy(h, s)

    def set_policy(self, h, s):
        occupancy = self.get_occupancy(h, s)
        state_occupancy = occupancy.sum()
        if state_occupancy > 0:
            self.policy[h][s] = occupancy / state_occupancy
        else:
            self.policy[h][s] = np.full(self.na, fill_value=1. / self.na)

    def get_occupancy(self, h, s):
        if self.solver == 'grb':
            occupancy = np.zeros(self.na, dtype=float)
            for a in self.actions:
                occupancy[a] = max(self.x[h, s, a].x, 0)
        else:
            occupancy = np.maximum(self.x[h][s].value, np.zeros(self.na))
        return occupancy

    def end_episode(self, evaluation=False):
        self.time_step = 0

    def expected_value(self, _) -> float:
        if self.solver == 'grb':
            return self.exp_reward.getValue()
        else:
            return self.exp_reward.value

    def get_expected_cost(self) -> float:
        if self.solver == 'grb':
            return self.exp_cost.getValue()
        else:
            return self.exp_cost.value

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
