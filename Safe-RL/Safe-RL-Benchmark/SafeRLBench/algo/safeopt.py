"""SafeOpt Wrapper."""

from SafeRLBench import AlgorithmBase
from SafeRLBench.error import add_dependency

from numpy import mean, array

try:
    import safeopt
except ModuleNotFoundError:
    safeopt = None

try:
    import GPy
except ModuleNotFoundError:
    GPy = None

import logging

logger = logging.getLogger(__name__)

__all__ = ('SafeOpt', 'SafeOptSwarm')


class _SafeOptWrap(AlgorithmBase):

    def __init__(self, opt, gp_opt_par, gp_par, environment, policy, max_it,
                 avg_reward, window):
        super(_SafeOptWrap, self).__init__(environment, policy, max_it)

        self._opt = opt

        self.gp_opt = None

        self.gp_opt_par = gp_opt_par
        self.gp_par = gp_par

        self.avg_reward = avg_reward
        self.window = window
        self.rewards = []

    def _initialize(self):
        logger.debug("Initializing Policy.")
        # check if policy is already initialized by the user
        if self.policy.initialized:
            logger.debug("Use pre-set policy parameters.")
            parameters = self.policy.parameters
        else:
            logger.debug("Draw parameters at random.")
            parameters = self.policy.parameter_space.sample()
            self.policy.parameters = parameters

        # Compute a rollout
        trace = self.environment.rollout(self.policy)
        reward = sum([t[2] for t in trace])

        # Initialize gaussian process with args:
        gp = []
        for pars in zip(*self.gp_par):
            gp.append(GPy.core.GP(array([parameters]), array([[reward]]),
                      *pars))

        # Initialize SafeOpt
        self.gp_opt = self._opt(gp, **self.gp_opt_par)

    def _step(self):
        parameters = self.gp_opt.optimize()
        self.policy.parameters = parameters

        trace = self.environment.rollout(self.policy)
        reward = sum([t[2] for t in trace])

        self.gp_opt.add_new_data_point(parameters, reward)
        self.rewards.append(reward)

    def _is_finished(self):
        if ((len(self.rewards) > self.window)
                and mean(self.rewards[(len(self.rewards) - self.window):-1])
                > self.avg_reward):
            return True
        else:
            return False


class SafeOpt(_SafeOptWrap):
    """Wrap SafeOpt algorithm.

    This class wraps the `SafeOpt` algorithm. It relies on the original
    implementation of `SafeOpt` which has to be installed before using this
    wrapper.

    Attributes
    ----------
    environment :
        Environment to be optimized.
    policy :
        Policy to be optimized.
    max_it :
        Maximal number of iterations before we abort.
    avg_reward : integer
        Average reward at which the optimization will be finished.
    window : integer
        Window for the average reward
    gp : GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    gp_opt : SafeOptSwarm object
        Instance of `SafeOptSwarm` used for optimization.
    gp_opt_par : dict
        Dictionary of parameters to initialize `SafeOpt`.
    """

    def __init__(self,
                 environment, policy, max_it, avg_reward, window,
                 kernel, likelihood, parameter_set, fmin,
                 lipschitz=None, beta=3.0, num_contexts=0, threshold=0,
                 scaling='auto', info=None):
        """Initialize Attributes.

        Parameters
        ----------
        environment :
            environmet to be optimized.
        policy :
            policy to be optimized.
        max_it :
            maximal number of iterations before we abort.
        avg_reward : integer
            average reward at which the optimization will be finished.
        window : integer
            window for the average reward
        kernel : GPy kernel
            Kernel used to initialize the gaussian process. If this is a list
            multiple kernels will be initialized. The size of this argument
            has to agree with the size of the likelihood.
        likelihood : GPy likelihood
            Likelihood used to initialize kernels. If this is a list, multiple
            kernels will be initialized. The size of this argument has to
            agree with the size of the likelihood.
        parameter_set : 2d-array
            List of parameters
        fmin : list of floats
            Safety threshold for the function value. If multiple safety
            constraints are used this can also be a list of floats (the first
            one is always the one for the values, can be set to None if not
            wanted)
        lipschitz : list of floats
            The Lipschitz constant of the system, if None the GP confidence
            intervals are used directly.
        beta : float or callable
            A constant or a function of the time step that scales the
            confidence interval of the acquisition function.
        threshold : float or list of floats
            The algorithm will not try to expand any points that are below this
            threshold. This makes the algorithm stop expanding points
            eventually. If a list, this represents the stopping criterion for
            all the gps. This ignores the scaling factor.
        scaling : list of floats or "auto"
            A list used to scale the GP uncertainties to compensate for
            different input sizes. This should be set to the maximal variance
            of each kernel. You should probably leave this to "auto" unless
            your kernel is non-stationary.
        info :
            Dummy argument that can hold anything usable to identify the
            configuration.
        """
        add_dependency(safeopt, 'SafeOpt')
        add_dependency(GPy, 'GPy')

        # store the `SafeOpt` arguments.
        gp_opt_par = {
            'parameter_set': parameter_set,
            'fmin': fmin,
            'lipschitz': lipschitz,
            'beta': beta,
            'num_contexts': num_contexts,
            'threshold': threshold,
            'scaling': scaling}

        # store the kernel arguments
        if not isinstance(kernel, list):
            kernel = [kernel]
        if not isinstance(likelihood, list):
            likelihood = [likelihood]
        assert len(likelihood) == len(kernel), (
            'kernel and likelihood need to have same length (%d /= %d)'
            % (len(likelihood), len(kernel)))

        gp_par = (kernel, likelihood)

        super(SafeOpt, self).__init__(safeopt.SafeOpt, gp_opt_par, gp_par,
                                      environment, policy, max_it, avg_reward,
                                      window)


class SafeOptSwarm(_SafeOptWrap):
    """Wrap SafeOpt algorithm.

    This class wraps the `SafeOptSwarm` algorithm. It relies on the original
    implementation of `SafeOptSwarm` which is part of the `safeopt` package
    and has to be installed before using this class.

    Attributes
    ----------
    environment :
        Environment to be optimized.
    policy :
        Policy to be optimized.
    max_it :
        Maximal number of iterations before we abort.
    avg_reward : integer
        Average reward at which the optimization will be finished.
    window : integer
        Window for the average reward
    gp : GPy Gaussian process
        A Gaussian process which is initialized with safe, initial data points.
        If a list of GPs then the first one is the value, while all the
        other ones are safety constraints.
    gp_opt : SafeOptSwarm object
        Instance of SafeOptSwarm used for optimization.
    gp_opt_par : list
        List of parameters to initialize `SafeOpt`.
    """

    def __init__(self,
                 environment, policy, max_it, avg_reward, window,
                 kernel, likelihood, fmin, bounds, beta=3.0, threshold=0,
                 scaling='auto', swarm_size=20, info=None):
        """Initialize Attributes.

        Parameters
        ----------
        environment :
            Environment to be optimized.
        policy :
            policy to be optimized.
        max_it :
            maximal number of iterations before we abort.
        avg_reward : integer
            average reward at which the optimization will be finished.
        window : integer
            window for the average reward
        kernel : GPy kernel
            Kernel used to initialize the gaussian process. If this is a list
            multiple kernels will be initialized. The size of this argument
            has to agree with the size of the likelihood.
        likelihood : GPy likelihood
            Likelihood used to initialize kernels. If this is a list, multiple
            kernels will be initialized. The size of this argument has to
            agree with the size of the likelihood.
        fmin : list of floats
            Safety threshold for the function value. If multiple safety
            constraints are used this can also be a list of floats (the first
            one is always the one for the values, can be set to None if not
            wanted)
        bounds : pair of floats or list of pairs of floats
            If a list is given, then each pair represents the lower/upper bound
            in each dimension. Otherwise, we assume the same bounds for all
            dimensions. This is mostly important for plotting or to restrict
            particles to a certain domain.
        beta : float or callable
            A constant or a function of the time step that scales the
            confidence interval of the acquisition function.
        threshold : float or list of floats
            The algorithm will not try to expand any points that are below this
            threshold. This makes the algorithm stop expanding points
            eventually. If a list, this represents the stopping criterion for
            all the gps. This ignores the scaling factor.
        scaling : list of floats or "auto"
            A list used to scale the GP uncertainties to compensate for
            different input sizes. This should be set to the maximal variance
            of each kernel. You should probably set this to "auto" unless your
            kernel is non-stationary
        swarm_size : int
            The number of particles in each of the optimization swarms
        info :
            Dummy argument that can hold anything usable to identify the
            configuration.
        """
        add_dependency(safeopt, 'SafeOpt')
        add_dependency(GPy, 'GPy')

        # store the `SafeOpt` arguments.
        gp_opt_par = {
            'fmin': fmin,
            'bounds': bounds,
            'beta': beta,
            'threshold': threshold,
            'scaling': scaling,
            'swarm_size': swarm_size
        }

        # store the kernel arguments
        if not isinstance(kernel, list):
            kernel = [kernel]
        if not isinstance(likelihood, list):
            likelihood = [likelihood]
        assert len(likelihood) == len(kernel), (
            'kernel and likelihood need to have same length (%d /= %d)'
            % (len(likelihood), len(kernel)))

        gp_par = (kernel, likelihood)

        super(SafeOptSwarm, self).__init__(safeopt.SafeOptSwarm, gp_opt_par,
                                           gp_par, environment, policy, max_it,
                                           avg_reward, window)
