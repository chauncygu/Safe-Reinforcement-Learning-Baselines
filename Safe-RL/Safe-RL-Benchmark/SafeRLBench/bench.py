"""Benchmarking facilities."""

from SafeRLBench import EnvironmentBase, AlgorithmBase
from SafeRLBench import config

from concurrent.futures import ProcessPoolExecutor

from itertools import product

import logging
import pprint
try:
    from string import maketrans
except ImportError:
    def maketrans(a, b):
        return str.maketrans(a, b)

logger = logging.getLogger(__name__)

__all__ = ('Bench', 'BenchConfig')


def _dispatch_wrap(run):
    return Bench._dispatch(run)


class Bench(object):
    """Benchmarking class to benchmark algorithms on various environments.

    This class needs a ``BenchConfig`` instance that provides it with the
    necessary configurations. Refer to the ``BenchConfig`` documentations for
    instruction of how to set up a configurations.

    Attributes
    ----------
    config :
        Instance of ``BenchConfig``
    runs : list
        List of ``BenchRun`` objects.
    measures : list
        List of measures

    Methods
    -------
    make_bench(algs, envs, measures=None)
        Static method that generates a configuration and returns an instance
        of Bench using this configuration.
    __call__()
        Benchmark then evaluate.
    benchmark()
        Initialize and run benchmark as configured.
    eval()
        Evaluate measures on test runs.

    Examples
    --------
    We can configure this class either in two steps, using the ``BenchConfig``
    class or in a single step using the ``make_bench`` method.

    Either way we start by defining our test runs. First define the environment
    configuration, e.g. LinearCar:

    >>> from SafeRLBench.envs import LinearCar
    >>> envs = [[(LinearCar, {'horizon': 100})]]

    Now we want to define a couple of different configurations for the
    PolicyGradient algorithm using list comprehension.

    >>> from SafeRLBench.algo import PolicyGradient
    >>> from SafeRLBench.policy import LinearPolicy
    >>> algs = [[
    ...     (PolicyGradient, [{
    ...         'policy': LinearPolicy(2, 1, par=[1, 1, 1]),
    ...         'estimator': 'central_fd',
    ...         'var': var
    ...     } for var in [1, 1.5, 2, 2.5]])
    ... ]]

    Now the first thing we could do is to manually generate a configuration and
    then use it to initialize ``Bench``.

    >>> from SafeRLBench import Bench, BenchConfig
    >>> config = BenchConfig(algs, envs)
    >>> bench = Bench(config)

    or we can do it in a single step using the static constructor in the
    ``Bench`` class.

    >>> # single step version is equivalent
    >>> bench = Bench.make_bench(algs, envs)

    To run the benchmarking and evaluation

    >>> bench()
    """

    def __init__(self, config=None, measures=None):
        """Initialize Bench instance.

        Parameters
        ----------
        configs : BenchConfig instance
            BenchConfig information supplying configurations. In case ``None``
            is passed an empty configuration will be instantiated. It can be
            populated with configurations using the ``add_tests`` methods of
            the configuration object. Default is ``None``.
        measures :
            List of measures from the measure module.
        """
        if not isinstance(config, BenchConfig):
            self.config = BenchConfig()
        else:
            self.config = config

        if measures is None:
            self.measures = []
        elif isinstance(measures, list):
            self.measures = measures
        else:
            self.measures = [measures]

        self.runs = []

    @staticmethod
    def make_bench(algs, envs, measures=None):
        """Construct a Bench directly.

        This will create the configuration instance and use it to create the
        bench. For detailed instructions on how the arguments work refer to
        the **Getting Started** section in the documentation or the
        ``BenchConfig`` documentation.

        Parameters
        ----------
        algs :
            List of list of tuples where the first element is an algorithm and
            the second a list of configurations. Any inner list may also be a
            single element instead of a list.
        envs :
            List of listof tuples where the first element is an environment and
            the second a configuration.
        measures :
            List of measures from the measure module.

        Returns
        -------
        bench : ``Bench`` object
            Confifigured ``Bench`` object.
        """
        config = BenchConfig(algs, envs)
        return Bench(config, measures)

    def __call__(self):
        """Initialize and run benchmark as configured."""
        self.benchmark()
        self.eval()

    def benchmark(self):
        """Initialize and run benchmark as configured."""
        logger.debug('Starting benchmarking.')

        self._set_up()

        if config.n_jobs > 1:
            self._benchmark_par()
        else:
            self._benchmark()

    def eval(self):
        """Evaluate measures on test runs."""
        for run in self.runs:
            if not run.completed:
                logger.warning("Evaluating before run completed.")

        for measure in self.measures:
            measure(self.runs)

    def _benchmark(self):
        for run in self.runs:
            self._dispatch(run)

    def _benchmark_par(self):
        n_jobs = config.n_jobs
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            fs = [ex.submit(_dispatch_wrap, run) for run in self.runs]
            self.runs = [f.result() for f in fs]

    def _set_up(self):
        self.runs = []
        for alg, env, alg_conf, env_conf in self.config:
            env_obj = env(**env_conf)
            alg_obj = alg(env_obj, **alg_conf)

            self.runs.append(BenchRun(alg_obj, env_obj, alg_conf, env_conf))

    @staticmethod
    def _dispatch(run):
        logger.debug('DISPATCH RUN:\n\n%s\n', str(run))

        run.alg.optimize()
        run.completed = True

        return run


class BenchConfig(object):
    """Benchmark configuration class.

    This class is supposed to provide a convenient interface to setup
    configurations for benchmarking runs.

    Attributes
    ----------
    algs :
        List of list of tuples where the first element is an algorithm and
        the second a list of configurations. Any inner list may also be a
        single element instead of a list.

    envs :
        List of list of tuples where the first element is an environment and
        the second a configuration. Any inner list may also be a single element
        instead of a list.

    Notes
    -----
    When defining configurations for benchmarking the algorithm's configuration
    may depend on the respective environments configuration. Thus using a
    simple cartesian product of configurations may not be reasonable in many
    cases.

    To provide an easy to use interface, we thus assume two mayor cases:
    * One environment configuration for multiple algorithm configurations.
    * One algorithm configuration for multiple environment configurations.

    The outer lists of the environment and algorithm configurations need to
    have equal size, because they represent reasonable configuration pairs,
    i.e. they will be zipped.
    From the pairs of inner lists we then generate a grid (cartesian product)
    of the configuration tuples, representing the two cases described above.

    See the examples sections for examples regarding the two cases.

    Examples
    --------
    Let's say we want to benchmark multiple algorithms on one reference
    environment.

    First define the environment configuration, e.g. LinearCar:

    >>> from SafeRLBench.envs import LinearCar
    >>> envs = [[(LinearCar, {'horizon': 100})]]

    Now we want to define a couple of different configurations for the
    PolicyGradient algorithm using list comprehension.

    >>> from SafeRLBench.algo import PolicyGradient
    >>> from SafeRLBench.policy import LinearPolicy
    >>> algs = [[
    ...     (PolicyGradient, [{
    ...         'policy': LinearPolicy(2, 1, par=[1, 1, 1]),
    ...         'estimator': 'central_fd',
    ...         'var': var
    ...     } for var in [1, 1.5, 2, 2.5]])
    ... ]]

    Last we can retrieve an instance and use it to initialize the ``Bench``.

    >>> from SafeRLBench import Bench, BenchConfig
    >>> config = BenchConfig(algs, envs)
    >>> bench = Bench(config)
    """

    def __init__(self, algs=None, envs=None):
        """Initialize BenchConfig instance.

        This initializer may be used if you want to test multiple algorithm
        configurations on a list of environments.

        Parameters
        ----------
        algs :
            List of list of tuples where the first element is an algorithm and
            the second a list of configurations. Any inner list may also be a
            single element instead of a list.
        envs :
            List of listof tuples where the first element is an environment and
            the second a configuration.
        """
        if algs is None or envs is None:
            self.algs = []
            self.envs = []
            return

        if (len(algs) != len(envs)):
            raise ValueError('Configuration lists dont have same length')

        for i, obj in enumerate(algs):
            algs[i] = self._listify(obj)

        for i, obj in enumerate(envs):
            envs[i] = self._listify(obj)

        self.algs = algs
        self.envs = envs

    def add_tests(self, algs, envs):
        """Add one set of environment and algorithm configurations.

        Parameters
        ----------
        algs :
            List of tuples where the first element is an algorithm and the
            second a list of configurations. May also be single elements.
        envs :
            List of tuple where the first element is an environment and the
            second a list of configurations. May also be single elements.
        """
        algs = self._listify(algs)
        envs = self._listify(envs)

        self.algs.append(algs)
        self.envs.append(envs)

    def _listify(self, obj):

        if not isinstance(obj, list):
            obj = [obj]

        for i, tup in enumerate(obj):

            if not isinstance(tup, tuple):
                raise ValueError('Invalid input structure.')

            try:
                if not issubclass(tup[0], (AlgorithmBase, EnvironmentBase)):
                    raise TypeError
            except TypeError:
                raise ValueError('Invalid Algorithm or Environment')

            if not isinstance(tup[1], list):
                obj[i] = (tup[0], [tup[1]])

        return obj

    def __iter__(self):
        for algs, envs in zip(self.algs, self.envs):
            for (alg, alg_confs), (env, env_confs) in product(algs, envs):
                for alg_conf, env_conf in product(alg_confs, env_confs):
                    yield alg, env, alg_conf, env_conf


class BenchRun(object):
    """Wrapper containing instances and configuration for a run.

    Attributes
    ----------
    alg :
        Algorithm instance
    env :
        Environment instance
    alg_conf : Dictionary
        Algorithm configuration
    env_conf : Dictionary
        Environment configuration

    Methods
    -------
    get_alg_monitor()
        Retrieve the data container from the algorithm instance.
    get_env_monitor()
        Retrieve the data container from the environment instance.
    """

    def __init__(self, alg, env, alg_conf, env_conf):
        """Initialize BenchRun instance.

        Parameters
        ----------
        alg :
            Algorithm instance
        env :
            Environment instance
        alg_conf : Dictionary
            Algorithm configuration
        env_conf : Dictionary
            Environment configuration
        """
        self.alg = alg
        self.env = env

        self.alg_conf = alg_conf
        self.env_conf = env_conf

        self.completed = False

    def get_alg_monitor(self):
        """Retrieve AlgMonitor for algorithm."""
        return self.alg.monitor

    def get_env_monitor(self):
        """Retrieve EnvMonitor for environment."""
        return self.env.monitor

    def __repr__(self):
        out = []
        out += ['Algorithm: ', [self.alg.__class__.__name__, self.alg_conf]]
        out += ['Environment: ', [self.env.__class__.__name__, self.env_conf]]

        return pprint.pformat(out, indent=2).translate(maketrans(',\'[]',
                                                                 '    '))
