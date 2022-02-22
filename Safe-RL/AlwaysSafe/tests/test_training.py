import unittest
import gym

import numpy as np
import matplotlib.pyplot as plt
from os import path

from planners import OptimisticLinearProgrammingPlanner
from util.mdp import monte_carlo_evaluation
from util.training import run_training_episodes
from util.training import save_and_plot
from util.training import train_agent


class TestTrainingAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("gym_factored:cliff_walking_cost-v0", num_rows=3, num_cols=4)
        cls.agent = OptimisticLinearProgrammingPlanner.from_discrete_env(cls.env, cost_bound=None, horizon=20)
        np.random.seed(42)
        cls.agent.solve()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree('/tmp/rluc/tests/')

    def test_train_optimal_agent(self):
        n = 4
        expected = {
            "training_returns": np.array([-5] * n),
            "training_costs": np.array([2] * n),
            "training_length": np.array([5] * n),
            "training_fail": np.array([0] * n),
            "evaluation_returns": np.array([-5] * n),
            "evaluation_costs": np.array([2] * n),
            "evaluation_length": np.array([5] * n),
            "evaluation_fail": np.array([0] * n),
        }
        results = run_training_episodes(self.agent, self.env, n, 20)
        for k in expected.keys():
            np.testing.assert_array_almost_equal(expected[k], results[k],
                                                 err_msg="{} is not equal".format(k))

    def test_evaluation_optimal_agent(self):
        ret, cost, length, fail = monte_carlo_evaluation(self.env, self.agent, self.agent.horizon, 1, 3)
        self.assertEqual(ret, -5)
        self.assertEqual(length, 5)
        self.assertEqual(cost, 2)
        self.assertEqual(fail, 0)

    def test_train_agent(self):
        train_agent(self.agent, self.env, 3, self.agent.horizon, 42, out_dir='/tmp/rluc/tests/training/', eval_episodes=10, verbose=False)
        self.assertTrue(path.exists('/tmp/rluc/tests/training/results_42.csv'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/training_returns_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/training_costs_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/training_length_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/training_fail_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/evaluation_returns_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/evaluation_costs_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/evaluation_length_42.png'))
        self.assertTrue(path.exists('/tmp/rluc/tests/training/evaluation_fail_42.png'))

    def test_plot_files(self):
        results = {"label":  np.random.random(10)}
        save_and_plot(results, '/tmp/rluc/tests/plotting', 42)
        self.assertTrue(path.exists('/tmp/rluc/tests/plotting/results_42.csv'))
        self.assertTrue(path.exists('/tmp/rluc/tests/plotting/label_42.png'))

    def test_plot_data(self):
        size = 100
        an_array = np.random.random(size)
        results = {"label": an_array}
        f, ax = plt.subplots()
        save_and_plot(results, '/tmp/rluc/tests/data', 42, testing=True)
        x_plot, y_plot = ax.lines[0].get_xydata().T
        np.testing.assert_array_almost_equal(y_plot, an_array)
        np.testing.assert_array_equal(x_plot, np.arange(size))

    def test_plot_data_rolling_average(self):
        size = 100
        window = 10
        an_array = np.ones(size)
        results = {"label": an_array}
        f, ax = plt.subplots()
        save_and_plot(results, '/tmp/rluc/tests/data2', 42, window=window, testing=True)
        x_plot, y_plot = ax.lines[1].get_xydata().T
        np.testing.assert_array_equal(x_plot, np.arange(size))
        np.testing.assert_array_almost_equal(y_plot, np.array([np.nan if i < window-1 else 1 for i in range(size)]))
