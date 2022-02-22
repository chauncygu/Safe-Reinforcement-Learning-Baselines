import unittest
import gym
import pybullet_envs  # noqa
import rl_safety_algorithms.common.utils as U
from rl_safety_algorithms.algs import core
import inspect
import sys
from rl_safety_algorithms.common.loggers import setup_logger_kwargs


class TestAlgorithms(unittest.TestCase):

    @staticmethod
    def check_alg(alg_name, env_id):
        """" Run one epoch update with algorithm."""
        print(f'Run {alg_name}.')
        defaults = U.get_defaults_kwargs(alg=alg_name, env_id=env_id)
        defaults['epochs'] = 1
        defaults['num_mini_batches'] = 4
        defaults['steps_per_epoch'] = 1000
        defaults['verbose'] = False

        defaults['logger_kwargs'] = setup_logger_kwargs(
            exp_name='unittest',
            seed=0,
            base_dir='/var/tmp/',
            datestamp=True,
            level=0,
            use_tensor_board=True,
            verbose=False)
        alg = U.get_alg_class(alg_name, env_id, **defaults)
        # sanity check of argument passing
        assert alg.alg == alg_name, f'Expected {alg_name} but got {alg.alg}'
        # return learn_fn(env_id, **defaults)
        ac, env = alg.learn()

        return ac, env

    def test_algorithms(self):
        """ Run all the specified algorithms."""
        algs = ['iwpg', 'npg', 'trpo', 'lag-trpo', 'pdo', 'cpo']
        for alg in algs:
            ac, env = self.check_alg(alg, 'HopperBulletEnv-v0')
            self.assertTrue(isinstance(env, gym.Env))


if __name__ == '__main__':
    unittest.main()
