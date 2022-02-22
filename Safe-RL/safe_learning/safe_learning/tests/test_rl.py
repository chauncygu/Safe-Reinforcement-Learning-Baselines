"""Unit tests for treinforcement learning."""

from __future__ import division, print_function, absolute_import

from numpy.testing import assert_allclose
import sys
import pytest
import tensorflow as tf
import numpy as np
import scipy.linalg
from safe_learning.utilities import dlqr

from safe_learning import (PolicyIteration, Triangulation, GridWorld,
                           QuadraticFunction, LinearSystem)

if sys.version_info.major <= 2:
    import mock
else:
    from unittest import mock

try:
    import cvxpy
except ImportError:
    cvxpy = None


class TestPolicyIteration(object):
    """Test the policy iteration."""
    def test_integration(self):
        """Test the values."""
        with tf.Session(graph=tf.Graph()) as sess:
            a = np.array([[1.2]])
            b = np.array([[0.9]])
            q = np.array([[1]])
            r = np.array([[0.1]])

            k, p = dlqr(a, b, q, r)
            true_value = QuadraticFunction(-p)

            discretization = GridWorld([[-1, 1]], 19)
            value_function = Triangulation(discretization,
                                           0. * discretization.all_points,
                                           project=True)

            dynamics = LinearSystem((a, b))

            policy_discretization = GridWorld([-1, 1], 5)
            policy = Triangulation(policy_discretization,
                                   -k / 2 * policy_discretization.all_points)
            reward_function = QuadraticFunction(-scipy.linalg.block_diag(q, r))

            rl = PolicyIteration(policy,
                                 dynamics,
                                 reward_function,
                                 value_function)

            value_iter = rl.value_iteration()

            loss = -tf.reduce_sum(rl.future_values(rl.state_space))
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            adapt_policy = optimizer.minimize(loss,
                                              var_list=rl.policy.parameters)

            sess.run(tf.global_variables_initializer())

            for _ in range(10):
                sess.run(value_iter)
                for _ in range(5):
                    sess.run(adapt_policy)

            values = rl.value_function.parameters[0].eval()
            true_values = true_value(rl.state_space).eval()
            policy_values = rl.policy.parameters[0].eval()

        assert_allclose(values, true_values, atol=0.1)
        assert_allclose(policy_values, -k * policy_discretization.all_points,
                        atol=0.1)
        #
        # assert(max_error < disc_error)
        # assert_allclose(rl.values, value_function.parameters[:, 0])

    @pytest.mark.skipif(cvxpy is None, reason='Cvxpy is not installed.')
    def test_optimization(self):
        """Test the value function optimization."""
        dynamics = mock.Mock()
        dynamics.return_value = np.arange(4, dtype=np.float)[:, None]

        rewards = mock.Mock()
        rewards.return_value = np.arange(4, dtype=np.float)[:, None]

        # transition probabilities
        trans_probs = np.array([[0, .5, .5, 0],
                                [.2, .1, .3, .5],
                                [.3, .2, .4, .1],
                                [0, 0, 0, 1]],
                               dtype=np.float)

        value_function = mock.Mock()
        value_function.tri.parameter_derivative.return_value = trans_probs
        value_function.nindex = 4
        value_function.parameters = [tf.Variable(np.zeros((4, 1),
                                                          dtype=np.float))]

        states = np.arange(4, dtype=np.float)[:, None]
        value_function.discretization.all_points = states

        policy = mock.Mock()
        policy.return_value = 'actions'

        rl = PolicyIteration(policy,
                             dynamics,
                             rewards,
                             value_function)

        true_values = np.linalg.solve(np.eye(4) - rl.gamma * trans_probs,
                                      rewards.return_value.ravel())[:, None]

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(value_function.parameters))
            sess.run(rl.optimize_value_function())
            values = rl.value_function.parameters[0].eval()

        # Confirm result
        assert_allclose(values, true_values)

        dynamics.assert_called_with(rl.state_space, 'actions')
        rewards.assert_called_with(rl.state_space, 'actions')

        # rl.terminal_states = np.array([0, 0, 0, 1], dtype=np.bool)
        # rl.optimize_value_function()
        #
        # trans_probs2 = np.array([[0, .5, .5, 0, 0],
        #                          [.2, .1, .3, .5, 0],
        #                          [.3, .2, .4, .1, 0],
        #                          [0, 0, 0, 0, 1],
        #                          [0, 0, 0, 0, 1]],
        #                         dtype=np.float)
        # rewards2 = np.zeros(5)
        # rewards2[:4] = rewards()
        # true_values = np.linalg.solve(np.eye(5) - rl.gamma * trans_probs2,
        #                               rewards2)
        #
        # assert_allclose(rl.values, true_values[:4])

    def test_future_values(self):
        """Test future values."""
        dynamics = mock.Mock()
        dynamics.return_value = 'next_states'

        rewards = mock.Mock()
        rewards.return_value = np.arange(4, dtype=np.float)[:, None]

        value_function = mock.Mock()
        value_function.return_value = np.arange(4, dtype=np.float)[:, None]
        value_function.discretization.all_points = \
            np.arange(4, dtype=np.float)[:, None]

        policy = mock.Mock()
        policy.return_value = 'actions'

        rl = PolicyIteration(policy,
                             dynamics,
                             rewards,
                             value_function)

        true_values = np.arange(4, dtype=np.float)[:, None] * (1 + rl.gamma)

        future_values = rl.future_values('states')

        dynamics.assert_called_with('states', 'actions')
        rewards.assert_called_with('states', 'actions')
        assert_allclose(future_values, true_values)

        # rl.terminal_states = np.array([0, 0, 0, 1], dtype=np.bool)
        # future_values = rl.get_future_values(rl.policy)
        # true_values[rl.terminal_states] = rewards()[rl.terminal_states]
        #
        # assert_allclose(future_values, true_values)


if __name__ == '__main__':
    pytest.main()
