""" Introduce an API which is similar to keras to train RL algorithms.

    Author:     Sven Gronauer
    Date:       19.05.2020
    Updated:    02.12.2020
"""
from rl_safety_algorithms.common.loggers import setup_logger_kwargs
from rl_safety_algorithms.common import utils
from rl_safety_algorithms.common import multi_processing_utils
import torch
import os
import gym


class Model(object):

    def __init__(self,
                 alg: str,
                 env_id: str,
                 log_dir: str,
                 init_seed: int,
                 unparsed_args: list = (),
                 use_mpi: bool = False,
                 ) -> None:
        """ Class Constructor  """
        self.alg = alg
        self.env_id = env_id
        self.log_dir = log_dir
        self.init_seed = init_seed
        # if MPI is not used, use Python's multi-processing
        self.multiple_individual_processes = False
        self.num_runs = 1
        self.num_cores = 1  # set by compile()-method
        self.training = False
        self.compiled = False
        self.trained = False
        self.use_mpi = use_mpi

        self.default_kwargs = utils.get_defaults_kwargs(alg=alg,
                                                        env_id=env_id)
        self.kwargs = self.default_kwargs.copy()
        self.kwargs['seed'] = init_seed
        # update algorithm kwargs with unparsed arguments from command line
        keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
        values = [eval(v) for v in unparsed_args[1::2]]
        unparsed_dict = {k: v for k, v in zip(keys, values)}
        self.kwargs.update(**unparsed_dict)
        self.logger_kwargs = None  # defined by compile (a specific seed might be passed)
        self.exp_name = os.path.join(self.env_id, self.alg)

        # assigned by class methods
        self.model = None
        self.env = None
        self.scheduler = None

    def _evaluate_model(self) -> None:
        from rl_safety_algorithms.common.experiment_analysis import EnvironmentEvaluator
        evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs['log_dir'])
        evaluator.eval(env=self.env, ac=self.model, num_evaluations=128)
        # Close opened files to avoid number of open files overflow
        evaluator.close()

    def _fill_scheduler(self, target_fn) -> None:
        """Create tasks for multi-process execution. This method is called when
        model.compile(individual_processes=True) is enabled.
        """
        ts = list()
        for task_number in range(1, self.num_runs + 1):
            kwargs = self.kwargs.copy()
            _seed = task_number + self.init_seed
            logger_kwargs = setup_logger_kwargs(base_dir=self.log_dir,
                                                exp_name=self.exp_name,
                                                seed=_seed,
                                                use_tensor_board=True,
                                                verbose=False)
            kwargs.update(logger_kwargs=logger_kwargs,
                          seed=_seed,
                          alg=self.alg,
                          env_id=self.env_id)
            t = multi_processing_utils.Task(
                id=_seed,
                target_function=target_fn,
                kwargs=kwargs)
            ts.append(t)
        self.scheduler.fill(tasks=ts)
        # adjust number of cores if num_runs < num_cores
        self.scheduler.num_cores = min(self.num_runs, self.scheduler.num_cores)

    @classmethod
    def _run_mp_training(cls, **kwargs):
        from rl_safety_algorithms.common.experiment_analysis import EnvironmentEvaluator
        alg = kwargs.pop('alg')
        env_id = kwargs.pop('env_id')
        logger_kwargs = kwargs.pop('logger_kwargs')

        learn_fn = utils.get_learn_function(alg)
        evaluator = EnvironmentEvaluator(log_dir=logger_kwargs['log_dir'])

        ac, env = learn_fn(
            env_id,
            logger_kwargs=logger_kwargs,
            **kwargs)
        evaluator.eval(env=env, ac=ac, num_evaluations=128)
        evaluator.close()

    def compile(self,
                num_runs=1,
                num_cores=os.cpu_count(),
                target='_run_mp_training',
                **kwargs_update
                ) -> object:
        """Compile the model.

        Either use mpi for parallel computation or run N individual processes.

        Parameters
        ----------
        num_runs
        num_cores
        use_mpi
        target
        kwargs_update

        Returns
        -------

        """
        # If MPI is not enabled, but the number of runs is greater than 1, then
        # start num_runs parallel processes, where each process is runs 
        # individually 
        if num_runs > 1 and not self.use_mpi:
            # First, reduce number of threads
            fair_num_threads = max(int(torch.get_num_threads() / num_runs), 1)
            torch.set_num_threads(fair_num_threads)
            self.num_runs = num_runs
            self.multiple_individual_processes = True
            self.scheduler = multi_processing_utils.Scheduler(num_cores=num_cores)
            target_fn = getattr(self, target)
            self._fill_scheduler(target_fn)

        self.kwargs.update(kwargs_update)
        _seed = self.kwargs.get('seed', self.init_seed)
        self.logger_kwargs = setup_logger_kwargs(base_dir=self.log_dir,
                                                 exp_name=self.exp_name,
                                                 seed=_seed)
        self.compiled = True
        self.num_cores = num_cores
        return self

    def _eval_once(self, actor_critic, env, render) -> tuple:
        done = False
        self.env.render() if render else None
        x = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            self.env.render() if render else None
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, info = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0)
            ret += r
            episode_length += 1
        return ret, episode_length, costs

    def eval(self, **kwargs) -> None:

        if self.multiple_individual_processes:
            # Note that Multi-process models are evaluated in _run_mp_training
            # method.
            pass
        else:
            self.model.eval()  # Set in evaluation mode before evaluation
            self._evaluate_model()
            self.model.train()  # switch back to train mode

    def fit(self, epochs=None, env=None) -> None:
        """ Train the model for a given number of epochs.

        Parameters
        ----------
        epochs: int
            Number of epoch to train. If None, use the standard setting from the
            defaults.py of the corresponding algorithm.
        env: gym.Env
            provide a custom environment for fitting the model, e.g. pass a
            virtual environment (based on NN approximation)

        Returns
        -------
        None

        """
        assert self.compiled, 'Call model.compile() before model.fit()'

        if self.multiple_individual_processes:
            # start all tasks which are stored in the scheduler
            self.scheduler.run()
        else:
            # single model training
            if epochs is None:
                epochs = self.kwargs.pop('epochs')
            else:
                self.kwargs.pop('epochs')  # pop to avoid double kwargs

            # fit() can also take a custom env, e.g. a virtual environment
            env_id = self.env_id if env is None else env

            learn_func = utils.get_learn_function(self.alg)
            ac, env = learn_func(
                env_id=env_id,
                logger_kwargs=self.logger_kwargs,
                epochs=epochs,
                **self.kwargs
            )
            self.model = ac
            self.env = env
        self.trained = True

    def play(self) -> None:
        """ Visualize model after training."""
        assert self.trained, 'Call model.fit() before model.play()'
        self.eval(episodes=5, render=True)

    def summary(self):
        """ print nice outputs to console."""
        raise NotImplementedError
