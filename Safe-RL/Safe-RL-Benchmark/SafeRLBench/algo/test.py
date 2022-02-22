"""Algorithm Tests."""

from SafeRLBench.algo import PolicyGradient, A3C
from SafeRLBench.envs import LinearCar
from .policygradient import CentralFDEstimator, estimators

from SafeRLBench.policy import NeuralNetwork

from unittest2 import TestCase
from mock import MagicMock, Mock


class TestPolicyGradient(TestCase):
    """PolicyGradientTestClass."""

    def test_pg_init(self):
        """Test: POLICYGRADIENT: initialization."""
        env_mock = MagicMock()
        pol_mock = Mock()

        for key, item in estimators.items():
            pg = PolicyGradient(env_mock, pol_mock, estimator=key)
            self.assertIsInstance(pg.estimator, item)

        pg = PolicyGradient(env_mock, pol_mock, estimator=CentralFDEstimator)
        self.assertIsInstance(pg.estimator, CentralFDEstimator)

        self.assertRaises(ImportError, PolicyGradient,
                          env_mock, pol_mock, CentralFDEstimator(env_mock))


class TestA3C(TestCase):
    """A3C Test Class."""

    def test_a3c_init(self):
        """Test: A3C: initialization."""
        a3c = A3C(LinearCar(), NeuralNetwork([2, 6, 1]))

        fields = ['environment', 'policy', 'max_it', 'num_workers', 'rate',
                  'done', 'policy', 'p_net', 'v_net', 'workers', 'threads',
                  'global_counter', 'sess']

        for field in fields:
            assert hasattr(a3c, field)
