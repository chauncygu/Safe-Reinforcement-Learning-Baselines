"""Linear Car."""
import numpy as np
from numpy import copy, array
from numpy.linalg import norm

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import RdSpace, BoundedSpace


# TODO: LinearCar: add examples
class LinearCar(EnvironmentBase):
    """Implementation of LinearCar Environment.

    This is a very simple environment implementing a car in an arbitrarily
    dimensioned space. By default it will just be one dimensional, which
    results in a two dimensional state space, that is, (pos, vel), and
    accordingly in a one dimensional bounded action space, that is, the
    acceleration.

    Attributes
    ----------
    state : ndarray
        Current state of the LinearCar.
    initial_state : ndarray
        Initial state of the LinearCar.
    goal : ndarray
        Goal state.
    eps : float
        Margin for completion. If 0, the goal is to stabilize at the goal
        completly.
    step : float
        Update step.
    state_space : Space object
        State space as deduced from the state.
    action_space : Space object
        Action space as deduced from the state.
    """

    def __init__(self, state=array([[0.], [0.]]), goal=array([[1.], [0.]]),
                 step=0.01, eps=0, horizon=100):
        """
        Initialize LinearCar.

        Parameters
        ----------
        state : ndarray
            Initial state of the LinearCar. The state and action space will be
            deduced from this. The shape needs to be (2, d) for d > 0.
        goal : ndarray
            Goal state of the LinearCar. The shape should comply to the shape
            of the initial state.
            In case the velocity is non-zero, eps should be strictly greater
            than zero, since there is no way for the system to stabilize in
            the goal state anyway.
        eps : float
            Reward at which we want to abort. If zero we do not abort at all.
        step : float
            Update step.
        """
        assert state.shape[0] == 2, 'Invalid shape of the initial state.'
        assert state.shape == goal.shape, 'State and goal shape have to agree.'

        # Initialize EnivronmentBase attributes
        self.horizon = horizon
        self.state_space = RdSpace(state.shape)
        self.action_space = BoundedSpace(-1, 1, shape=(state.shape[1],))

        # Initialize State
        self.initial_state = state
        self.state = copy(state)

        # Initialize Environment Parameters
        self.goal = goal
        self.eps = eps
        self.step = step

    def _update(self, action):
        one = np.ones(self.action_space.shape)
        action = np.maximum(np.minimum(action, one), -one)

        self.state[1] += self.step * action
        self.state[0] += self.state[1]

        return (action, copy(self.state), self._reward())

    def _reset(self):
        self.state = copy(self.initial_state)

    def _rollout(self, policy):
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
            if (self.eps != 0 and self._achieved()):
                return trace
        return trace

    def _reward(self):
        return -norm(self.state - self.goal)

    def _achieved(self):
        return (abs(self._reward()) < self.eps)
