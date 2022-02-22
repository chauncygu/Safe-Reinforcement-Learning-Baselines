import numpy as np
from typing import Sequence

from gym_factored.envs.base import DiscreteEnv

from planners.abs_lp_optimistic import AbsOptimisticLinearProgrammingPlanner
from util.mdp import get_mdp_functions
from util.mdp import get_mdp_functions_partial

np.seterr(invalid='ignore', divide='ignore')


class AbsOptCMDPAgent:
    def __init__(self,
                 ns: int,
                 na: int,
                 terminal: np.array,
                 isd: np.array,
                 env,
                 max_reward, min_reward,
                 abs_transition, abs_cost, abs_terminal, abs_map,
                 horizon=3,
                 cost_bound=None,
                 policy_type='ground',
                 cost_bound_coefficient=1,
                 solver='grb',  # grb, cvxpy
                 verbose=False):
        self.ns, self.na = ns, na
        self.terminal = terminal
        self.isd = isd
        self.env = env
        self.horizon = horizon
        self.cost_bound = cost_bound
        self.policy_type = policy_type
        self.cost_bound_coefficient = cost_bound_coefficient
        self.verbose = verbose

        self.max_reward = max(max_reward, 0)
        self.min_reward = min_reward
        if terminal.any():
            self.max_reward = max(self.max_reward, 0)

        self.abs_transition = abs_transition
        self.abs_cost = abs_cost
        self.abs_terminal = abs_terminal
        self.abs_map = abs_map

        self.estimated_transition = np.full((ns, na, ns), fill_value=1/ns)
        self.estimated_reward = np.full((ns, na), fill_value=self.max_reward)
        self.ensure_terminal_states_are_absorbing()

        self.counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.new_counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.acc_reward = np.zeros((ns, na))

        self.solver = solver
        self.planner = self.instantiate_planner()

        # computing initial policy
        self.planner.solve()

    def instantiate_planner(self):
        if (self.counter_sas > 0).any():
            inverse_counter = 1 / np.maximum(self.counter_sas.sum(axis=2), 1)
            var_transition = self.estimated_transition * (1 - self.estimated_transition)

            transition_ci = np.sqrt(var_transition * inverse_counter[:, :, np.newaxis]) + inverse_counter[:, :, np.newaxis]
            transition_ci[self.terminal] = 0

            if False:
                # this is the theoretical upper-bound on the confidence interval
                reward_ci = np.sqrt(inverse_counter) * (self.max_reward - self.min_reward)
            else:
                reward_ci = inverse_counter * (self.max_reward - self.min_reward)
            reward_ci[self.terminal] = 0
        else:
            transition_ci=np.full((self.ns, self.na, self.ns), fill_value=1.0)
            reward_ci=np.full((self.ns, self.na), fill_value=self.max_reward - self.min_reward)

        return AbsOptimisticLinearProgrammingPlanner(
            self.estimated_transition, self.estimated_reward, None, self.terminal, self.isd,
            self.env, self.max_reward, self.min_reward, 0, 0,
            abs_transition=self.abs_transition,
            abs_cost=self.abs_cost,
            abs_terminal=self.abs_terminal,
            abs_map=self.abs_map,
            cost_bound=self.cost_bound,
            horizon=self.horizon,
            transition_ci=transition_ci,
            reward_ci=reward_ci,
            verbose=self.verbose,
            policy_type=self.policy_type,
            cost_bound_coefficient=self.cost_bound_coefficient,
            solver=self.solver
        )

    @classmethod
    def from_discrete_env(cls, env: DiscreteEnv, features: Sequence = None, **kwargs) -> 'AbsOptCMDPAgent':
        transition, reward, _, terminal = get_mdp_functions(env)
        for s in np.arange(env.nS)[terminal]:
            transition[s, :, :] = 0
            transition[s, :, s] = 1
            reward[s, :] = 0
        max_reward, min_reward = reward.max(), reward.min()
        abs_transition, abs_reward, abs_cost, abs_terminal, abs_map = get_mdp_functions_partial(env, features)
        for s in np.arange(abs_terminal.shape[0])[abs_terminal]:
            abs_transition[s, :, :] = 0
            abs_transition[s, :, s] = 1

        return cls(env.nS, env.nA, terminal, env.isd, env, max_reward, min_reward,
                   abs_transition, abs_cost, abs_terminal, abs_map, **kwargs)

    def ensure_terminal_states_are_absorbing(self):
        for s in np.arange(self.ns)[self.terminal]:
            self.estimated_transition[s, :, :] = 0
            self.estimated_transition[s, :, s] = 1
            self.estimated_reward[s, :] = 0

    def act(self, state):
        return self.planner.act(state)

    def add_transition(self, state, reward, action, next_state, done, info=None):
        self.acc_reward[state, action] += reward
        self.new_counter_sas[state, action, next_state] += 1
        self.planner.add_transition(state, reward, action, next_state, done, info)

    def end_episode(self, evaluation=False):
        if not evaluation:
            if self.enough_new_samples_collected():
                self.aggregate_new_samples()
                self.update_estimate()
                self.update_planner()
                self.planner.solve()
        self.planner.end_episode()

    def enough_new_samples_collected(self):
        return (self.new_counter_sas > self.counter_sas).any()

    def aggregate_new_samples(self):
        self.counter_sas += self.new_counter_sas
        self.new_counter_sas.fill(0)

    def update_estimate(self):
        counter_sa = np.maximum(self.counter_sas.sum(axis=2), 1)
        self.estimated_transition = self.counter_sas/counter_sa[:, :, np.newaxis]
        self.estimated_reward = self.acc_reward/counter_sa
        self.ensure_terminal_states_are_absorbing()

    def update_planner(self):
        self.planner = self.instantiate_planner()

    def expected_value(self, _) -> float:
        return self.planner.expected_value(self.isd)

    def get_expected_cost(self) -> float:
        return self.planner.get_expected_cost()

    def seed(self, seed):
        self.planner.seed(seed)

