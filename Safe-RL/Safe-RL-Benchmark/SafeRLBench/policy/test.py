"""Policy tests."""
from __future__ import division, print_function, absolute_import

from SafeRLBench.spaces import BoundedSpace
from SafeRLBench.envs.quadrocopter import Reference
from SafeRLBench.envs._quadrocopter import StateVector
from SafeRLBench.policy import (NeuralNetwork,
                                LinearPolicy,
                                DiscreteLinearPolicy,
                                NonLinearQuadrocopterController)

import numpy as np
from numpy import isclose

import tensorflow as tf

from unittest2 import TestCase
from mock import Mock

import logging

logger = logging.getLogger(__name__)


class TestNeuralNetwork(TestCase):
    """Test the Neural Netork Policy."""

    fields = ['args', 'kwargs', 'action_space', 'state_space', 'dtype',
              'layers', 'scope', 'init_weights', 'activation', 'X', 'a',
              'W_action', 'W_var', 'a_pred', 'var', 'h', 'is_set_up']

    def test_initialization(self):
        """Test: NEURALNETWORK: initialization."""
        # test bad layer size:
        args = [[2]]
        with self.assertRaises(ValueError):
            NeuralNetwork(*args)

        # test field existence
        args = [[2, 6, 1]]

        nn = NeuralNetwork(*args)

        for field in self.fields:
            assert hasattr(nn, field)

        # test network setup
        kwargs = {
            'do_setup': True
        }

        nn = NeuralNetwork(*args, **kwargs)

        # check field contents.
        assert(all([a == b for a, b in zip(args, nn.args)]))
        self.assertEqual(nn.layers, args[0])
        self.assertEqual(nn.dtype, 'float')

        self.assertEqual(len(nn.W_action), 2)
        self.assertEqual(len(nn.W_var), 1)

        # well... because is does not work for whatever fucking reason.
        self.assertEqual(str(type(nn.a_pred)), str(tf.Tensor))
        self.assertIn(str(type(nn.var)), (str(tf.Tensor), str(tf.constant)))

        self.assertEqual(len(nn.h), 2)

    def test_mapping(self):
        """Test: NEURALNETWORK: mapping."""
        args = [[2, 1]]

        kwargs = {
            'weights': [tf.constant([2., 1.], shape=(2, 1))],
            'do_setup': True,
        }

        nn = NeuralNetwork(*args, **kwargs)

        sess = tf.Session()

        with sess.as_default():
            self.assertEqual(nn(np.array([2., 1.])), [5.])

    def test_variable_assignment(self):
        """Test: NEURALNETWORK: parameter assignment."""
        args = [[2, 1]]
        kwargs = {'do_setup': True}

        nn = NeuralNetwork(*args, **kwargs)

        with tf.Session().as_default():
            nn.parameters = nn.W_action[0].assign([[2.], [1.]])
            assert((np.array([[2.], [1.]]) == nn.parameters).all())
            self.assertEqual(nn(np.array([2., 1.])), [5.])

    def test_copy(self):
        """Test: NEURALNETWORK: copy."""
        nn = NeuralNetwork([2, 6, 1])
        nn_copy = nn.copy(scope='copy', do_setup=False)

        exclude = ('scope', 'kwargs')

        for field in self.fields:
            if field not in exclude and field in nn.kwargs.keys():
                print(field)
                self.assertEquals(getattr(nn, field, None),
                                  getattr(nn_copy, field, None))


class TestLinearPolicy(TestCase):
    """Test the Linear Policy."""

    def test_initialization(self):
        """Test: LINEARPOLICY: initialization."""
        lp = LinearPolicy(2, 1)

        self.assertEqual(lp.d_state, 2)
        self.assertEqual(lp.d_action, 1)

        self.assertEqual(lp.par_dim, 2)
        self.assertIs(lp._par_space, None)

        self.assertFalse(lp.initialized)

        self.assertIs(lp._parameters, None)
        self.assertTrue(lp.biased)
        self.assertEqual(lp._bias, 0)
        self.assertIs(lp._par, None)

        par_mock = Mock()
        par_space_mock = Mock()

        with self.assertRaises(ValueError):
            lp_mocked = LinearPolicy(2, 1, par_mock, par_space_mock)

        par_mock = [2, 1]

        lp_mocked = LinearPolicy(2, 1, par_mock, par_space_mock)

        self.assertTrue(lp_mocked.initialized)
        assert(all(par_mock == lp_mocked.parameters))

        self.assertEqual(par_space_mock, lp_mocked.parameter_space)

    def test_discrete_map(self):
        """Test: DISCRETELINEARPOLICY: map."""
        dp = DiscreteLinearPolicy(2, 1, biased=False)
        dp.parameters = np.array([1, 1])
        self.assertEqual(dp([1, 1]), 1)
        self.assertEqual(dp([-1, -1]), 0)

        dp2 = DiscreteLinearPolicy(2, 2, biased=False)
        dp2.parameters = np.array([1, 1, -1, -1])
        assert(all(dp2([1, 1]) == [1, 0]))
        assert(all(dp2([-1, -1]) == [0, 1]))


class TestController(TestCase):
    """Test NonLinearQuadrocopterController."""

    def test_controller_init(self):
        """Test: CONTROLLER: initialization."""
        ctrl = NonLinearQuadrocopterController()

        self.assertEquals(ctrl._zeta_z, .7)
        assert(all(isclose(ctrl._params, [.7, .7, .7, .5, .707])))
        self.assertIsNone(ctrl.reference)
        self.assertTrue(ctrl.initialized)
        self.assertIsInstance(ctrl._par_space, BoundedSpace)

    def test_controller_map(self):
        """Test: CONTROLLER: mapping."""
        ref = Reference('circle', 1 / 70.)
        ref.reset(StateVector())
        ctrl = NonLinearQuadrocopterController(reference=ref)

        action = ctrl(StateVector())

        print(action)
        assert all(isclose(action, [0.20510876, -0.30667618, 0., -6.28318548]))

    def test_controller_properties(self):
        """Test: CONTROLLER: properties."""
        ctrl = NonLinearQuadrocopterController()

        ctrl.parameters = [0., 1., 0., 1., 0.]
        assert(all(np.isclose(ctrl.parameters, [0., 1., 0., 1., 0.])))

        self.assertEquals(ctrl.parameter_space, ctrl._par_space)
