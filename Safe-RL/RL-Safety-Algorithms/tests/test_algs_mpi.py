import unittest
import gym
import pybullet_envs  # noqa
import time
import rl_safety_algorithms.common.mpi_tools as mpi_tools
from rl_safety_algorithms.common.loggers import setup_logger_kwargs
import rl_safety_algorithms.common.utils as U
import warnings

warnings.filterwarnings("ignore")


class TestAlgorithms(unittest.TestCase):

    @staticmethod
    def check_alg(alg_name, env_id, cores):
        """" Run one epoch update with algorithm."""
        defaults = U.get_defaults_kwargs(alg=alg_name, env_id=env_id)
        defaults['epochs'] = 1
        defaults['num_mini_batches'] = 4
        defaults['steps_per_epoch'] = 1000 * mpi_tools.num_procs()
        defaults['verbose'] = False
        print(defaults['steps_per_epoch'])

        defaults['logger_kwargs'] = setup_logger_kwargs(
            exp_name='unittest',
            seed=0,
            base_dir='/var/tmp/',
            datestamp=True,
            level=0,
            use_tensor_board=False,
            verbose=False)
        alg = U.get_alg_class(alg_name, env_id, **defaults)
        # sanity check of argument passing
        assert alg.alg == alg_name, f'Expected {alg_name} but got {alg.alg}'
        # return learn_fn(env_id, **defaults)
        ac, env = alg.learn()
        return ac, env

    def test_mpi_version_of_algorithms(self):
        """ Run all the specified algorithms with MPI."""
        cores = 4
        if mpi_tools.mpi_fork(n=cores):  # forks the current script and use MPI
            return  # use return instead of sys.exit() to exit test with 'OK'...
        is_root = mpi_tools.proc_id() == 0
        algs = ['iwpg', 'npg', 'trpo', 'lag-trpo', 'pdo', 'cpo']
        for alg in algs:
            try:
                print(f'Run {alg.upper()}') if is_root else None
                ac, env = self.check_alg(alg,
                                         'HopperBulletEnv-v0',
                                         cores=cores)
                self.assertTrue(isinstance(env, gym.Env))
            except NotImplementedError:
                print('No MPI yet supported...') if is_root else None
        else:
            # sleep one sec to finish all console prints...
            time.sleep(1)


if __name__ == '__main__':
    unittest.main()
