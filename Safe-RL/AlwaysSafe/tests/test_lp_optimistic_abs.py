import unittest
import gym

import numpy as np

from planners import AbsOptimisticLinearProgrammingPlanner
from util.mdp import monte_carlo_evaluation

np.set_printoptions(precision=3, suppress=True)


def _solve(env, cost_bound, horizon=20):
    lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(env, [0, 1], cost_bound=cost_bound, horizon=horizon)
    np.random.seed(42)
    lp_agent.solve()
    expected_value = lp_agent.expected_value(env.isd)
    expected_cost = lp_agent.get_expected_cost()
    expected_value_mc, expected_cost_mc, _, _ = monte_carlo_evaluation(env, lp_agent, horizon)
    # assert abs(expected_value_mc - expected_value) < 0.5, "{:0.3f} != {:0.3f}".format(expected_value_mc, expected_value)
    assert abs(expected_cost_mc - expected_cost) < 0.5, "{:0.3f} != {:0.3f}".format(expected_cost_mc, expected_cost)
    return expected_value, expected_cost


class TestLPAgentChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:chain2d-v0")

    def test_lp_agent_horizon_zero(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=0)
        self.assertAlmostEqual(expected_value, 0, places=2)
        self.assertAlmostEqual(expected_cost, 0)

    def test_lp_agent_horizon_one(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=1)
        self.assertAlmostEqual(expected_value, 4, places=2)
        self.assertAlmostEqual(expected_cost, 1)

    def test_lp_agent_horizon_two(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=2)
        self.assertAlmostEqual(expected_value, 8.25, places=2)
        self.assertAlmostEqual(expected_cost, 1.75)

    def test_lp_agent_unbounded(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=3)
        self.assertAlmostEqual(expected_value, 10, places=2)
        self.assertAlmostEqual(expected_cost, 2)

    def test_lp_agent_bounded(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=0, horizon=3)
        self.assertAlmostEqual(expected_value, 1, places=2)
        self.assertAlmostEqual(expected_cost, 0, places=2)


class TestLPAgentCliff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0", num_rows=3, num_cols=4)

    def test_lp_agent_unbounded(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=None)
        self.assertAlmostEqual(expected_value, -5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)

    def test_lp_agent_bounded_0(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=0)
        self.assertAlmostEqual(expected_value, -7, places=2)
        self.assertAlmostEqual(expected_cost, 0, places=2)

    def test_lp_agent_bounded_1(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=1)
        self.assertAlmostEqual(expected_value, -6, places=2)
        self.assertAlmostEqual(expected_cost, 1, places=2)

    def test_lp_agent_bounded_2(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=2)
        self.assertAlmostEqual(expected_value, -5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)


@unittest.skip("skip slow test")
class TestLPAgentCliffLarge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0")

    def test_lp_agent_unbounded(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=None, horizon=20)
        self.assertAlmostEqual(expected_value, -13, places=2)
        self.assertAlmostEqual(expected_cost, 20, places=2)

    def test_lp_agent_bounded(self):
        expected_value, expected_cost = _solve(self.env, cost_bound=0, horizon=20)
        self.assertAlmostEqual(expected_value, -17, places=2)
        self.assertAlmostEqual(expected_cost, 0, places=2)


class TestLPAgentCliffConfidenceInterval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0", num_rows=3, num_cols=4)

    def test_reward_ci(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [0, 1], cost_bound=None, horizon=20,
            reward_ci=np.full((self.env.nS, self.env.nA), fill_value=0.5),
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_value, -2.5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)

    def test_transition_ci(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [0, 1], cost_bound=None, horizon=20,
            transition_ci=np.full((self.env.nS, self.env.nA, self.env.nS), fill_value=0.1),
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()

        self.assertAlmostEqual(expected_value, -5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)

    def test_transition_large_ci(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [0, 1], cost_bound=None, horizon=20,
            transition_ci=np.full((self.env.nS, self.env.nA, self.env.nS), fill_value=1),
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()

        self.assertAlmostEqual(expected_value, -5, places=2)
        self.assertAlmostEqual(expected_cost, 2, places=2)


class TestLPAgentSimpleCMDP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cmdp-v0")

    def test_unbounded(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [0], cost_bound=None, horizon=6
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_value, 12, places=2)
        self.assertAlmostEqual(expected_cost, 6, places=2)

    def test_bounded(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [0], cost_bound=3, horizon=6
        )
        # np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_cost, 3, places=2)
        self.assertAlmostEqual(expected_value, 6.66, places=1)

    def test_unbounded_abs(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [], cost_bound=None, horizon=6
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_value, 12, places=2)
        self.assertAlmostEqual(expected_cost, 6, places=2)

    def test_bounded_abs(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [], cost_bound=3, horizon=6
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_cost, 3, places=2)
        self.assertAlmostEqual(expected_value, 6.66, places=1)

    def test_unbounded_abs_unknown_reward_transition(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [], cost_bound=None, horizon=6,
            reward_ci=np.full((self.env.nS, self.env.nA), fill_value=2),
            transition_ci=np.full((self.env.nS, self.env.nA, self.env.nS), fill_value=1),
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_value, 18, places=2)
        self.assertAlmostEqual(expected_cost, 6, places=2)

    def test_bounded_abs_unknown_reward_transition(self):
        lp_agent = AbsOptimisticLinearProgrammingPlanner.from_discrete_env(
            self.env, [], cost_bound=3, horizon=6,
            reward_ci=np.full((self.env.nS, self.env.nA), fill_value=3),
            transition_ci=np.full((self.env.nS, self.env.nA, self.env.nS), fill_value=1),
        )
        np.random.seed(42)
        lp_agent.solve()
        expected_value = lp_agent.expected_value(self.env.isd)
        expected_cost = lp_agent.get_expected_cost()
        self.assertAlmostEqual(expected_value, 18, places=2)
        self.assertLessEqual(expected_cost, 3)

        # _, expected_cost_mc, _, _ = monte_carlo_evaluation(self.env, lp_agent, 6, number_of_episodes=2000)
        # self.assertLessEqual(expected_cost_mc, 3.1)
