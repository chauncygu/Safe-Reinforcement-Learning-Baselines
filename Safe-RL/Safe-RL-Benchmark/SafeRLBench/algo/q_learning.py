"""Q-learning implementations."""

from SafeRLBench import AlgorithmBase, Policy
from SafeRLBench.spaces import DiscreteSpace
from SafeRLBench.error import IncompatibilityException

import numpy as np


# TODO: DiscreteQLearning: examples, monitoring, finished, adaptive rate
class DiscreteQLearning(AlgorithmBase):
    """Q-Learning Algorithm.

    This Algorithm estimates a quality measure that maps every (state, action)
    pair to a real number.

    Attributes
    ----------
    Q : ndarray
        Array representing the quality for each state action pair.
    environment :
        The environment for which we want to estimate the Q function. Its
        state and action space need to be an instance of `DiscreteSpace`.
    discount : float
        Discount factor.
    max_it : int
        Maximum number of iterations.
    rate : float
        Update rate.
    shape : (int, int)
        Tuple containing the dimension of the state and action space.

    Notes
    -----
    The environment needs to use a discrete state and action space, because
    this Q-Learning implementation uses a table to estimate the Q function.
    """

    def __init__(self, environment, discount, max_it, rate):
        """Initialize QLearning.

        Parameters
        ----------
        environment :
            The environment for which we want to estimate the Q function. Its
            state and action space need to be an instance of `DiscreteSpace`.
        discount : float
            Discount factor.
        max_it : int
            Maximum number of iterations.
        rate : float
            Update rate.
        """
        # make some sanity checks
        if (not isinstance(self.environment.action_space, DiscreteSpace)
           and not isinstance(self.environment.state_space, DiscreteSpace)):
            raise IncompatibilityException(self, self.environment)

        if discount <= 0:
            raise ValueError('discount %d needs to be larger than zero.',
                             discount)

        if max_it <= 0:
            raise ValueError('max_it %d needs to be larger than zero.', max_it)

        # initialize the fields
        self.environment = environment
        self.discount = discount
        self.max_it = max_it
        self.rate = rate

        # determine the dimension of the state and action space
        d_state = environment.state_space.dimension
        d_action = environment.action_space.dimension

        self.shape = (d_state, d_action)

        # initialize the lookup table for the Q function.
        self.Q = None
        self.policy = _RandomPolicy(environment.action_space)

    def _initialize(self):
        self.Q = np.zeros(self.shape)

    def _step(self):
        trace = self.environment.rollout(self.policy)
        for (action, state, reward) in trace:
            dq = (reward + self.discount * self.Q[state, :].max()
                  - self.Q[state, action])
            self.Q[state, action] += self.rate * dq

    def _is_finished(self):
        pass

    # TODO: Q-learning evaluate qlearning performance appropiately


class _RandomPolicy(Policy):

    def __init__(self, action_space):
        self.action_space = action_space

    def map(self, state):
        return self.action_space.sample

    @property
    def parameters(self):
        return self.action_space.dimension

    @property
    def parameter_space(self):
        return None
