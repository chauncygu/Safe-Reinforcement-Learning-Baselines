""" Benchmark different algorithms and setups on environments.

    Author: Sven Gronauer
    Date:   27.10.2020

    ----------
    Usage, e.g.

    alg_setup = {
        'trpo': {"target_kl": [0.01, 0.02, 0.03]},
        'cpo': {'target_kl': [0.01, 0.02, 0.03], 'lam_c': [1.0, 0.9, 0.5]}
    }
    bench = Benchmark(
        alg_setup,
        env_id='SafetyAntRun-v0',
        log_dir='/var/tmp',
        num_cores=4,
        num_runs=2,
    )
    bench.run()
"""

import sys
import os
import warnings
import json
from itertools import product
import psutil
# === RL-Safety-Algorithm packages
from rl_safety_algorithms.common.multi_processing_utils import Scheduler, Task
import rl_safety_algorithms.common.mpi_tools as mpi_tools
from rl_safety_algorithms.common.loggers import setup_logger_kwargs
from rl_safety_algorithms.common import utils


try:
    import pybullet_envs  # noqa
except ImportError:
    warnings.warn('pybullet_envs package not found.')
try:
    import bullet_safety_gym  # noqa
except ImportError:
    warnings.warn('bullet-safety-gym package not found.')


def run_training(**kwargs) -> None:
    from rl_safety_algorithms.common.experiment_analysis import EnvironmentEvaluator
    alg = kwargs.pop('alg')
    env_id = kwargs.pop('env_id')
    logger_kwargs = kwargs.pop('logger_kwargs')
    evaluator = EnvironmentEvaluator(
        log_dir=logger_kwargs['log_dir'],
        log_costs=True)
    learn = utils.get_learn_function(alg)
    ac, env = learn(env_id,
                    logger_kwargs=logger_kwargs,
                    **kwargs)
    evaluator.eval(
        ac=ac,
        env=env,
        num_evaluations=128)
    # close output files after evaluation to limit number of open files
    evaluator.close()


class Benchmark:
    """ Benchmark several algorithms on certain environments.

        important input paramater:

        alg_setup: dict
            {
                'trpo': {"target_kl": [0.01, 0.001], "gamma": [0.95, 0.9]}
            }
    """

    def __init__(self,
                 alg_setup: dict,
                 env_ids: list,
                 log_dir: str,
                 num_cores: int,
                 num_runs: int,
                 env_specific_kwargs: dict,
                 use_mpi: bool = True,
                 init_seed: int = 0
                 ) -> None:
        self.env_ids = env_ids
        self.env_specific_kwargs = env_specific_kwargs
        self.log_dir = log_dir
        self.alg_setup = alg_setup
        self.init_seed = init_seed
        self.num_cores = num_cores
        self.num_runs = num_runs
        self.use_mpi = use_mpi
        if not use_mpi:
            # Scheduler holds queue with parameters for individual processes
            self.scheduler = Scheduler(num_cores=num_cores, verbose=False)
            self._fill_scheduler()
        else:
            # re-start the current script
            self.scheduler = None
            physical_cores = psutil.cpu_count(logical=False)  # exclude hyper-threading
            # Use number of physical cores as default. If also hardware
            # threading CPUs should be used, enable this by:
            use_num_of_threads = True if num_cores > physical_cores else False
            if mpi_tools.mpi_fork(num_cores,
                                  use_number_of_threads=use_num_of_threads):
                sys.exit()

    @classmethod
    def _convert_to_dict(cls, param_grid) -> dict:
        # convert string to dict
        if isinstance(param_grid, str):
            param_grid = json.loads(param_grid)
        elif isinstance(param_grid, dict):
            pass
        else:
            raise TypeError(f'param_grid of type: {type(param_grid)}')
        return param_grid

    def _fill_scheduler(self):
        """ Create tasks for multi-process execution.

        will be called if model.compile(multi_thread=True) is enabled."""
        ts = list()
        _seed = self.init_seed

        for env_id in self.env_ids:
            for alg_name, param_grid in self.alg_setup.items():
                param_grid = self._convert_to_dict(param_grid)
                exp_name = os.path.join(env_id, alg_name)

                for param_set in product(*param_grid.values()):
                    grid_kwargs = dict(zip(param_grid.keys(), param_set))

                    for i in range(self.num_runs):
                        kwargs = utils.get_defaults_kwargs(
                            alg=alg_name,
                            env_id=env_id
                        )
                        logger_kwargs = setup_logger_kwargs(
                            base_dir=self.log_dir,
                            exp_name=exp_name,
                            seed=_seed,
                            level=0,
                            use_tensor_board=True,
                            verbose=False)
                        kwargs.update(logger_kwargs=logger_kwargs,
                                      seed=_seed,
                                      alg=alg_name,
                                      env_id=env_id)
                        # firstly, update environment specifics
                        kwargs.update(**self.env_specific_kwargs[env_id])
                        # secondly, pass the grid search parameters...
                        kwargs.update(**grid_kwargs)

                        t = Task(
                            id=_seed,
                            target_function=run_training,
                            kwargs=kwargs)
                        ts.append(t)
                        _seed += 1
        self.scheduler.fill(tasks=ts)
        # reduce number of cores if necessary
        self.scheduler.num_cores = min(_seed, self.num_cores)

    def mpi_run(self):
        """Run parameter grid over all MPI processes. No scheduling required."""
        init_seed = self.init_seed
        for env_id in self.env_ids:
            for alg_name, param_grid in self.alg_setup.items():
                param_grid = self._convert_to_dict(param_grid)
                exp_name = os.path.join(env_id, alg_name)

                for param_set in product(*param_grid.values()):
                    grid_kwargs = dict(zip(param_grid.keys(), param_set))

                    for i in range(self.num_runs):
                        if mpi_tools.is_root_process():
                            print(f'Run #{i} (with seed={init_seed}) and kwargs:')
                            print(grid_kwargs)

                        kwargs = utils.get_defaults_kwargs(
                            alg=alg_name,
                            env_id=env_id
                        )
                        logger_kwargs = setup_logger_kwargs(
                            base_dir=self.log_dir,
                            exp_name=exp_name,
                            seed=init_seed,
                            level=0,
                            use_tensor_board=True,
                            verbose=False)
                        kwargs.update(logger_kwargs=logger_kwargs,
                                      seed=init_seed,
                                      alg=alg_name,
                                      env_id=env_id)
                        # firstly, update environment specifics
                        kwargs.update(**self.env_specific_kwargs[env_id])
                        # secondly, pass the grid search parameters...
                        kwargs.update(**grid_kwargs)
                        run_training(**kwargs)
                        init_seed += 1

    def run(self):
        if self.use_mpi:
            self.mpi_run()
        else:
            # Kick-off scheduled tasks
            self.scheduler.run()
