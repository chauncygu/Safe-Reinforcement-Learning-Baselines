"""General Mountain Car."""
import numpy as np
from numpy import pi, array, copy, cos, sin

from SafeRLBench.base import EnvironmentBase
from SafeRLBench.spaces import BoundedSpace


class GeneralMountainCar(EnvironmentBase):
    """Implementation of a GeneralMountainCar Environment.

    Attributes
    ----------
    state_space : BoundedSpace
        Space object describing the state space.
    action_space : BoundedSpace
        Space object describing the action space.
    state : array-like
        Current state of the car.
    initial_state : array-like
        Initial state of the car.
    gravitation : double
    power : double
    goal : double
        Goal along x-coordinate
    """

    def __init__(self,
                 state_space=BoundedSpace(array([-1, -0.07]),
                                          array([1, 0.07])),
                 action_space=BoundedSpace(-1, 1, shape=(1,)),
                 state=np.array([0, 0]),
                 contour=None, gravitation=0.0025, power=0.0015,
                 goal=0.6, horizon=100):
        """Initialize GeneralMountainCar Environment.

        Parameters
        ----------
        state_space : BoundedSpace
            Space object describing the state space.
        action_space : BoundedSpace
            Space object describing the action space.
        state : array-like
            Initial state of the car
        contour : tuple of callables
            If contour is None, a default shape will be generated. A valid
            tuple needs to contain a function for the height at a position
            in the first element and a function for the gradient at a position
            in the second argument.
        gravitation : double
        power : double
        goal : double
            Goal along x-coordinate
        """
        # Initialize Environment Base Parameters
        super(GeneralMountainCar, self).__init__(state_space,
                                                 action_space,
                                                 horizon)

        # setup environment parameters
        self.goal = goal
        self.power = power
        self.gravitation = gravitation

        # setup contour
        if contour is None:
            def _hx(x):
                return -cos(pi * x)
            self._hx = _hx

            def _dydx(x):
                return pi * sin(pi * x)
            self._dydx = _dydx
        else:
            self._hx = contour[0]
            self._dydx = contour[1]

        # init state
        self.state = copy(state)
        self.initial_state = state

    def _update(self, action):
        """Compute step considering the action."""
        action = array(action).flatten()
        action = max(min(action, 1.0), -1.0)

        if hasattr(action, 'size') and action.size == 1:
            action_in = action[0]
        else:
            action_in = action

        position = self.state[0]
        velocity = self.state[1]

        velocity += (action_in * self.power
                     - self._dydx(position) * self.gravitation)
        position += velocity

        bounds = self.state_space

        velocity = max(min(velocity, bounds.upper[1]), bounds.lower[1])
        position = max(min(position, bounds.upper[0]), bounds.lower[0])

        # make sure outputs have the right form
        self.state = np.array([position, velocity])
        action = np.reshape(action, self.action_space.shape)

        return action, copy(self.state), self._reward()

    def _reset(self):
        self.state = copy(self.initial_state)

    def _reward(self):
        return(self.height() - 1)

    def _rollout(self, policy):
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
            if (self.position() >= self.goal):
                return trace
        return trace

    def height(self):
        """Compute current height."""
        return(self._hx(self.state[0].item()).item())

    def position(self):
        """Compute current position in x."""
        return(self.state[0])
