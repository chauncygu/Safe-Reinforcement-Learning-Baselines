from SafeRLBench import config

from SafeRLBench import Bench, BenchConfig
from SafeRLBench.algo import PolicyGradient
from SafeRLBench.envs import LinearCar
from SafeRLBench.policy import LinearPolicy
from SafeRLBench.measure import BestPerformance


from unittest2 import TestCase

import logging

logger = logging.getLogger(__name__)


class TestIntegration(TestCase):
    """Test integration with PolicyGradient and LinearCar."""

    def test_integration(self):
        """Integration: bench with pc and lc."""
        # setup config:
        config.logger_set_level(logging.DEBUG)
        config.monitor_set_verbosity(3)

        policy = LinearPolicy(2, 1, biased=True)
        algs = [(PolicyGradient, {'policy': policy,
                                  'max_it': 10,
                                  'estimator': 'central_fd'})]
        env = [[(LinearCar, {'horizon': 100})]]

        test_config = BenchConfig(algs, env)

        benchmark = Bench(test_config, [BestPerformance()])
        benchmark()

        assert(benchmark.measures[0].result is not None)

    def test_parallel_integration(self):
        """Integration: bench with pc and lc (parallel)."""
        # setup config:
        config.logger_set_level(logging.DEBUG)
        config.monitor_set_verbosity(3)
        config.jobs_set(2)

        policy = LinearPolicy(2, 1)
        algs = [(PolicyGradient, [{'policy': policy,
                                   'max_it': 10,
                                   'estimator': 'central_fd'},
                                  {'policy': policy,
                                   'max_it': 20,
                                   'estimator': 'central_fd'}])]
        env = [[(LinearCar, {'horizon': 100})]]

        test_config = BenchConfig(algs, env)

        benchmark = Bench(test_config, [BestPerformance()])
        benchmark()

        assert(benchmark.measures[0].result is not None)
        assert(len(benchmark.measures[0].result) == 2)
