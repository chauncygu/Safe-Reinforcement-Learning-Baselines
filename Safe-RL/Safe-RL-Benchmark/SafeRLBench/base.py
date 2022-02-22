"""Module implements Baseclasses."""

from __future__ import division, print_function, absolute_import

from abc import ABCMeta, abstractmethod
from six import add_metaclass

from SafeRLBench import AlgoMonitor, EnvMonitor

__all__ = ('EnvironmentBase', 'Space')


@add_metaclass(ABCMeta)
class EnvironmentBase(EnvMonitor):
    """Environment Base Class.

    This base class defines and implements an interface to any environment
    implementation part of the environment module. Subclasses inheriting
    from EnvironmentBase need to make sure they meet the requirements below.

    Any subclass must implement:
        * _update(action)
        * _reset()

    Any subclass might override:
        * _rollout(policy)

    Make sure the `state_space`, `action_space` and `horizon` attributes will
    be set in any subclass, as the default implementation and / or the monitor
    may access them to retrieve information.

    Attributes
    ----------
    state_space :
        State space of the environment.
    action_space :
        Action space of the environment.
    horizon :
        Maximum number of iterations until rollout will stop.
    monitor : EnvData instance
        Contains the monitoring data. The monitor will be automatically
        initialized during creation.

    Methods
    -------
    rollout(policy)
        Perform a rollout according to the actions selected by policy.
    update(action)
        Update the environment state according to the action.
    reset()
        Reset the environment to the initial state.

    Notes
    -----
    When overwriting _rollout(policy) use the provided interface functions
    and do not directly call the private implementation.
    """

    def __init__(self, state_space, action_space, horizon=0):
        """Initialize EnvironmentBase.

        Parameters
        ----------
        state_space :
            State space of the environment.
        action_space :
            Action space of the environment.
        horizon :
            Maximum number of iterations until rollout will stop.
        """
        super(EnvironmentBase, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.horizon = horizon

    # Implement in subclasses:
    # See update(self, action) for more information
    @abstractmethod
    def _update(self, action):
        raise NotImplementedError

    # See reset(self) for more information
    @abstractmethod
    def _reset(self):
        raise NotImplementedError

    # Override in subclasses if necessary
    def _rollout(self, policy):
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
        return trace

    def update(self, action):
        """Update the environment state according to the action.

        Wraps the subclass implementation _update(action) providing
        monitoring capabilities.

        Parameters
        ----------
        action: array-like
            Element of action_space

        Returns
        -------
        tuple : 3-tuple
            action : array-like
                element of action space as it has been applied in update
            state : array-like
                element of state_space which is the resulting state after
                applying action
            reward : float
                reward for resulting state
        """
        with self.monitor_update():
            t = self._update(action)
        return t

    def reset(self):
        """Reset the environment to initial state.

        Reset wraps the subclass implementation _reset() providing monitoring
        capabilities.
        """
        with self.monitor_reset():
            self._reset()

    def rollout(self, policy):
        """Perform a rollout according to the actions selected by policy.

        Wraps the implementation _rollout(policy) providing monitoring
        capabilities.

        Parameters
        ----------
        Policy : callable
            Maps element of state_space to element of action_space

        Returns
        -------
        trace : list of 3-tuple
            List of (action, state, reward)-tuple as returned by update().
        """
        with self.monitor_rollout():
            trace = self._rollout(policy)
        return trace

    def __repr__(self):
        """Return class name."""
        return self.__class__.__name__


@add_metaclass(ABCMeta)
class Space(object):
    """Baseclass for Spaceobject.

    All methods have to be implemented in any subclass.

    Methods
    -------
    contains(x)
        Check if x is an element of space.
    element
        Return arbitray element in space.
    """

    @abstractmethod
    def contains(self, x):
        """Check if x is an element of space."""
        pass

    @abstractmethod
    def sample(self):
        """Return an arbitrary element in space for unit testing."""
        pass

    @property
    @abstractmethod
    def dimension(self):
        """Return the dimension of the space."""
        pass


@add_metaclass(ABCMeta)
class AlgorithmBase(AlgoMonitor):
    """Baseclass for any algorithm.

    This baseclass defines a uniform interface for any algorithm part of
    the algorithm module SafeRLBench.algo. It features monitoring capabilities
    for tracking and evaluating the execution of the algorithm.

    Inheriting from `AlgorithmBase` is suspect to some constraints, i.e. any
    algorithm needs to be implemented using the following functions.

    Any subclass must overwrite:
        * _initialize(policy)
        * _step(policy)
        * _is_finished()

    Any subclass may overwrite:
        * _optimize(policy)

    In case one does overwrite _optimize, the functions _initialize(),
    _step(parameter), _is_finished() may just pass unless they are used.
    This may however change the information tracked by the monitor.

    Attributes
    ----------
    environment :
        Environment we want to optimize on
    policy :
        Policy to be optimized
    max_it : int
        Maximum number of iterations
    monitor : AlgoData instance
        Contains monitoring data. The monitor will automatically initialize
        on creation of an object.

    Methods
    -------
    optimize()
        Optimize a policy with respective algorithm.
    initialize()
        Initialize policy parameter.
    step()
        Update policy parameters.
    is_finished()
        Return true when algorithm is finished.

    Notes
    -----
    Specification of the private functions.

    _initialize(self):
        Initialize the algorithm.
    _step():
        Compute one step of the algorithm.
    _is_finished():
        Return True when algorithm is supposed to finish.
    """

    def __init__(self, environment, policy, max_it):
        super(AlgorithmBase, self).__init__()

        self.environment = environment
        self.policy = policy
        self.max_it = max_it

        self.grad = None

    # Have to be overwritten.
    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _step(self):
        pass

    @abstractmethod
    def _is_finished(self):
        pass

    # May be overwritten
    def _optimize(self):
        self.initialize()

        for n in range(self.max_it):
            self.step()
            if self.is_finished():
                break

    def optimize(self):
        """Optimize policy parameter.

        Wraps subclass implementation in _optimize(policy).

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        with self.monitor_optimize():
            self._optimize()

    def initialize(self):
        """Initialize policy parameter.

        Wraps subclass implementation in _initialize(policy)

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        with self.monitor_initialize():
            self._initialize()

    def step(self):
        """Update policy parameter.

        Wraps subclass implementation in _step(policy).

        Parameters
        ----------
        policy: PolicyBase subclass
        """
        with self.monitor_step():
            self._step()

    def is_finished(self):
        """Return True when algorithm is supposed to finish.

        Wraps subclass implementation in _is_finished().
        """
        stop = self._is_finished()
        return stop

    def reset(self):
        """Reset the monitor."""
        self._alg_reset()

    def __repr__(self):
        if hasattr(self, '_info'):
            return self._info()
        return self.__class__.__name__


@add_metaclass(ABCMeta)
class Policy(object):
    """Minimal policy interface."""

    def __call__(self, state):
        return self.map(state)

    @abstractmethod
    def map(self, state):
        """Map element of state space to action space."""
        pass

    @property
    @abstractmethod
    def parameters(self):
        """Access current parameters."""
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, par):
        pass

    @property
    @abstractmethod
    def parameter_space(self):
        """Return parameter space."""


@add_metaclass(ABCMeta)
class ProbPolicy(Policy):
    """Probabilistic policy interface."""

    @abstractmethod
    def grad_log_prob(self, state, action):
        """Return the :math:log(grad p(action | state)):math:."""
        pass
