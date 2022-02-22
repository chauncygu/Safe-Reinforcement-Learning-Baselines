"""Test the utilities."""

from __future__ import absolute_import, print_function, division

import pytest
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose

from safe_learning.utilities import (dlqr, get_storage, set_storage,
                                     get_feed_dict, unique_rows,
                                     compute_trajectory)

from safe_learning import LinearSystem


def test_dlqr():
    """Test the dlqr function."""
    true_k = np.array([[0.61803399]])
    true_p = np.array([[1.61803399]])

    k, p = dlqr(1, 1, 1, 1)
    assert_allclose(k, true_k)
    assert_allclose(p, true_p)

    k, p = dlqr([[1]], [[1]], [[1]], [[1]])
    assert_allclose(k, true_k)
    assert_allclose(p, true_p)


class TestStorage(object):
    """Test the class storage."""

    @pytest.fixture
    def sample_class(self):
        """Sample class for testing."""
        class A(object):
            """Some class."""

            def __init__(self):
                """Initialize."""
                super(A, self).__init__()
                self.storage = {}

            def method(self, value, index=None):
                storage = get_storage(self.storage, index=index)
                set_storage(self.storage, [('value', value)], index=index)
                return storage

        return A()

    def test_storage(self, sample_class):
        """Test the storage."""
        storage = sample_class.method(5)
        assert storage is None
        storage = sample_class.method(4)
        assert storage['value'] == 5
        storage = sample_class.method(None)
        assert storage['value'] == 4

        # Test index
        storage = sample_class.method(3, index='test')
        assert storage is None
        storage = sample_class.method(4, index='test')
        assert storage['value'] == 3
        storage = sample_class.method(3, index='test2')
        assert storage is None
        storage = sample_class.method(3, index='test')
        assert storage['value'] is 4


def test_get_feed_dict():
    """Test the global get_feed_dict method."""
    graph = tf.Graph()
    feed_dict = get_feed_dict(graph)
    # Initialized new dictionary
    assert feed_dict == {}

    # Test assignment
    feed_dict['test'] = 5

    # Make sure we keep getting the same object
    assert feed_dict is get_feed_dict(graph)


def test_unique_rows():
    """Test the unique_rows function."""
    a = np.array([[1, 1], [1, 2], [1, 3], [1, 2], [1, 3], [1, 4], [2, 3]])
    uniques = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [2, 3]])

    assert_allclose(unique_rows(a), uniques)


def test_compute_trajectory():
    """Test the compute_trajectory function."""
    A = np.array([[1., 0.1],
                  [0., 1.]])
    B = np.array([[0.01],
                  [0.1]])

    dynamics = LinearSystem((A, B))
    Q = np.diag([1., 0.01])
    R = np.array([[0.01]])
    K, _ = dlqr(A, B, Q, R)
    policy = LinearSystem([-K])

    x0 = np.array([[0.1, 0.]])
    with tf.Session() as sess:
        res = compute_trajectory(dynamics, policy, x0, num_steps=20)

    states, actions = res
    assert_allclose(states[[0], :], x0)
    assert_allclose(states[-1, :], np.array([0., 0.]), atol=0.01)
    assert_allclose(actions, states[:-1].dot(-K.T))