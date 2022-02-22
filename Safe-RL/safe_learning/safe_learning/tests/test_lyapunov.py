"""Unit tests for the Lyapunov functions."""

from __future__ import division, print_function, absolute_import

from numpy.testing import assert_allclose, assert_equal
import pytest
import unittest
import numpy as np
import tensorflow as tf
import sys

from safe_learning.functions import (LinearSystem, GridWorld)
from safe_learning.lyapunov import (Lyapunov, smallest_boundary_value)

if sys.version_info.major <= 2:
    import mock
else:
    from unittest import mock


class TestLyapunov(object):
    """Test the Lyapunov base class."""

    def test_safe_set_init(self):
        """Test the safe set initialization."""
        with tf.Session():
            discretization = GridWorld([[0, 1], [0, 1]], 3)
            lyap_fun = lambda x: tf.reduce_sum(tf.square(x), axis=1)

            dynamics = LinearSystem(np.array([[1, 0.01],
                                              [0., 1.]]))
            lf = 0.4
            lv = 0.3
            eps = 0.5

            policy = lambda x: 0. * x
            lyap = Lyapunov(discretization, lyap_fun, dynamics, lf, lv,
                            eps, policy)

            initial_set = [1, 3]
            lyap = Lyapunov(discretization, lyap_fun, dynamics, lf, lv,
                            eps, policy, initial_set=initial_set)

            initial_set = np.array([False, True, False, True, False,
                                    False, False, False, False])
            assert_equal(initial_set, lyap.safe_set)

    def test_update(self):
        """Test the update step."""
        with tf.Session():
            discretization = GridWorld([[-1, 1]], 3)
            lyap_fun = lambda x: tf.reduce_sum(tf.square(x),
                                               axis=1,
                                               keep_dims=True)
            policy = lambda x: -.1 * x

            dynamics = LinearSystem(np.array([[1, 1.]]))
            lf = 0.4
            lv = 0.3
            eps = .5

            initial_set = [1]

            lyap = Lyapunov(discretization, lyap_fun, dynamics, lf, lv,
                            eps, policy, initial_set=initial_set)

            lyap.update_safe_set()
            assert_equal(lyap.safe_set, np.array([False, True, False]))

            eps = 0.
            lyap = Lyapunov(discretization, lyap_fun, dynamics, lf, lv,
                            eps, policy, initial_set=initial_set)
            lyap.update_safe_set()
            assert_equal(lyap.safe_set, np.ones(3, dtype=np.bool))


def test_smallest_boundary_value():
    """Test the boundary value function."""
    with tf.Session():
        fun = lambda x: 2 * tf.reduce_sum(tf.abs(x), axis=1)
        discretization = GridWorld([[-1.5, 1], [-1, 1.5]], [3, 3])
        min_value = smallest_boundary_value(fun, discretization)
        assert min_value == 2.5


if __name__ == '__main__':
    unittest.main()
