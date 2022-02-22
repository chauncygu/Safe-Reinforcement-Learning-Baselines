import numpy as np

from gym_factored.envs.base import DiscreteEnv

from planners.lp_optimistic import OptimisticLinearProgrammingPlanner
from util.mdp import get_mdp_functions

np.seterr(invalid='ignore', divide='ignore')


class OptCMDPAgent:
    def __init__(self,
                 ns: int,
                 na: int,
                 terminal: np.array,
                 isd: np.array,
                 env,
                 max_reward, min_reward,
                 max_cost, min_cost,
                 horizon=3,
                 cost_bound=None,
                 solver='grb',  # grb, cvxpy
                 verbose=False):
        self.ns, self.na = ns, na
        self.terminal = terminal
        self.isd = isd
        self.env = env
        self.horizon = horizon
        self.cost_bound = cost_bound
        self.verbose = verbose

        self.max_reward = max(max_reward, 0)
        self.min_reward = min_reward
        self.max_cost = max_cost
        self.min_cost = min(min_cost, 0)
        if terminal.any():
            self.max_reward = max(self.max_reward, 0)
            self.min_cost = min(self.min_cost, 0)

        self.estimated_transition = np.full((ns, na, ns), fill_value=1/ns)
        self.estimated_reward = np.full((ns, na), fill_value=self.max_reward)
        self.estimated_cost = np.full((ns, na), fill_value=self.min_cost)
        self.ensure_terminal_states_are_absorbing()

        self.counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.new_counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.acc_reward = np.zeros((ns, na))
        self.acc_cost = np.zeros((ns, na))

        self.solver = solver

        self.planner = OptimisticLinearProgrammingPlanner(
            self.estimated_transition, self.estimated_reward, self.estimated_cost, self.terminal, self.isd,
            self.env, self.max_reward, self.min_reward, self.max_cost, self.min_cost,
            cost_bound=self.cost_bound, horizon=self.horizon,
            transition_ci=np.full((self.ns, self.na, self.ns), fill_value=1.0),
            reward_ci=np.full((self.ns, self.na), fill_value=self.max_reward-self.min_reward),
            cost_ci=np.full((self.ns, self.na), fill_value=self.max_cost-self.min_cost),
            solver=self.solver,
            verbose=self.verbose
        )

        # computing initial policy
        self.planner.solve()

    @classmethod
    def from_discrete_env(cls, env: DiscreteEnv, **kwargs) -> 'OptCMDPAgent':
        _, reward, cost, terminal = get_mdp_functions(env)
        max_reward, min_reward = reward.max(), reward.min()
        max_cost, min_cost = cost.max(), cost.min()
        return cls(env.nS, env.nA, terminal, env.isd, env, max_reward, min_reward, max_cost, min_cost, **kwargs)

    def ensure_terminal_states_are_absorbing(self):
        for s in np.arange(self.ns)[self.terminal]:
            self.estimated_transition[s, :, :] = 0
            self.estimated_transition[s, :, s] = 1
            self.estimated_reward[s, :] = 0
            self.estimated_cost[s, :] = 0

    def act(self, state):
        return self.planner.act(state)

    def add_transition(self, state, reward, action, next_state, done, info=None):
        self.acc_reward[state, action] += reward
        self.acc_cost[state, action] += info.get('cost', 0)
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
        self.estimated_cost = self.acc_cost/counter_sa
        self.ensure_terminal_states_are_absorbing()

    def update_planner(self):

        inverse_counter = 1 / np.maximum(self.counter_sas.sum(axis=2), 1)
        var_transition = self.estimated_transition * (1 - self.estimated_transition)

        transition_ci = np.sqrt(var_transition * inverse_counter[:, :, np.newaxis]) + inverse_counter[:, :, np.newaxis]
        transition_ci[self.terminal] = 0

        # this is the theoretical upper-bound on the confidence interval
        # reward_ci = np.sqrt(inverse_counter) * (self.max_reward - self.min_reward)
        # cost_ci = np.sqrt(inverse_counter) * (self.max_cost - self.min_cost)

        # we use a tighter upper-bound to reduce the computational burden
        reward_ci = inverse_counter * (self.max_reward - self.min_reward)
        cost_ci = inverse_counter * (self.max_cost - self.min_cost)

        reward_ci[self.terminal] = 0
        cost_ci[self.terminal] = 0

        self.planner = OptimisticLinearProgrammingPlanner(
            self.estimated_transition, self.estimated_reward, self.estimated_cost, self.terminal, self.isd,
            self.env, self.max_reward, self.min_reward, self.max_cost, self.min_cost,
            cost_bound=self.cost_bound, horizon=self.horizon,
            transition_ci=transition_ci,
            reward_ci=reward_ci,
            cost_ci=cost_ci,
            solver=self.solver,
            verbose=self.verbose
        )

    def expected_value(self, _) -> float:
        return self.planner.expected_value(self.isd)

    def get_expected_cost(self) -> float:
        return self.planner.get_expected_cost()

    def seed(self, seed):
        self.planner.seed(seed)
