"""Quadrocopter environment wrapper."""

from __future__ import division, print_function, absolute_import

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import RdSpace

from ._quadrocopter import QuadrotorDynamics
from ._quadrocopter import StateVector

from functools import partial

from six import string_types

import numpy as np
from numpy import array
from numpy import pi, cos, sin
from numpy.linalg import norm

import logging

logger = logging.getLogger(__name__)

# Available reference functions.
REFERENCE_TYPES = ['circle', 'stationary', 'oscillate']


class Quadrocopter(EnvironmentBase):
    """Quadrocopter simulation.

    Attributes
    ----------
    horizon : int
        Number of iterations for the main simulation
    pre_sim_horizon : int
        Number of iterations for the pre-simulation.
    _model : model object
        Object simulating the quadrotor dynamics.
    """

    def __init__(self,
                 init_pos=None, init_vel=None, num_sec=9,
                 num_init_sec=4, ref='circle', period=1 / 70.,
                 seed=None):
        """Quadrocopter initialization.

        Parameters
        ----------
        init_pos : array-like
            Initial position of the quadrocopter. Default: None ; which will
            set init_pos to [1, 0, 0].
        init_vel : array-like
            Initial velocity of the quadrocopter. Default: None ; which will
            set init_vel to [0, pi / 2, 0]
        num_sec : integer
        num_init_sec : integer
        ref : string or reference object
            Name of the reference. Currently supported are 'circle',
            'stationary' or 'oscillate'.
        period : float
        seed : int
        """
        # spaces
        self.state_space = RdSpace((22,))
        self.action_space = RdSpace((4,))

        # seed
        if seed is not None:
            np.random.seed = seed
            self._seed = seed

        # initial position
        if init_pos is None:
            init_pos = array([cos(0), sin(0), 0.])

        if len(init_pos) != 3:
            raise ValueError("init_pos with invalid length %d.", init_pos)

        # initial velocity
        if init_vel is None:
            init_vel = array([-pi / 2. * sin(0), pi / 2. * cos(0), 0.])

        if len(init_vel) != 3:
            raise ValueError("init_vel with invalid length %d.", init_vel)

        # initialize model
        self._model = QuadrotorDynamics(init_pos, init_vel)

        if isinstance(ref, string_types):
            self.reference = Reference(ref, period)
        else:
            self.reference = ref
            self.period = ref.period

        self.reference.reset(self.state)

        self.horizon = int(1. / period) * num_sec
        self.pre_sim_horizon = int(1. / period) * num_init_sec

        self.period = self.reference.period

        self._init_pos = init_pos
        self._init_vel = init_vel

        self._trajectory = np.atleast_2d(np.zeros(3))
        self._time = []
        self._step = 0

    def _update(self, action):
        assert self.action_space.contains(action), "Invalid action."

        self._model.update_position(action)

        self._step += 1
        time = self._step * self.period

        self._time.append(time)
        self._trajectory = np.vstack((self._trajectory, self.state.pos))

        reward = self._reward()
        self.reference.update(self.state, time)

        return action, self.state.copy(), reward

    def _reset(self):
        self._model = QuadrotorDynamics(self._init_pos, self._init_vel)
        self.reference.reset(self.state)
        self._trajectory = np.atleast_2d(np.zeros(3))
        self._time = []
        self._step = 0

    def _rollout(self, policy):
        if hasattr(policy, 'reference'):
            policy.reference = self.reference
        self.reset()
        trace = []
        for n in range(self.horizon):
            action = policy(self.state)
            trace.append(self.update(action))
        return trace

    def _reward(self):
        state = self.state
        ref = self.reference.reference

        reward = -norm(state.pos - ref.pos) - norm(state.vel - ref.vel)

        if np.isnan(reward):
            reward = -1.79769313e+308

        return reward

    @property
    def seed(self):
        """Seed."""
        return self._seed

    @seed.setter
    def seed(self, value):
        np.random.seed(value)
        self._seed = value

    @property
    def state(self):
        """Provide access to state_vector."""
        # this whole state vector implementation is annoyingly inefficient.
        return self._model.state.state_vector

    @state.setter
    def state(self, state):
        self._model.state.state_vector = state.view(StateVector)


class Reference(object):
    """Reference object for quadrocopter environment."""

    def __init__(self, name='circle', period=1. / 70, keep_record=True,
                 **kwargs):
        """Initialize Reference.

        Parameters
        ----------
        name : str
            The name of the reference function.
        period : float
            The time step that the simulation takes every iteration.
        keep_record : bool
            Whether the history of the reference object should be saved.
        **kwargs : dict
        """
        # name type checking.
        if not isinstance(name, string_types):
            raise ValueError('Invalid type for argument name.')
        if name not in REFERENCE_TYPES:
            raise ValueError(name + ' is not a valid reference.')

        self._name = name
        self.period = period
        self._iter = 0
        self._reference_function = self._reference_chooser(**kwargs)
        self._current_ref = None
        self.keep_record = keep_record
        if keep_record:
            self._record = []

    @property
    def name(self):
        """Return the type of reference function."""
        return self._name

    @name.setter
    def name(self, value):
        if value not in REFERENCE_TYPES:
            raise ValueError(value + ' is not a valid reference.')

        self.reset()

        self._name = value
        self._reference_function = self._ref_chooser()

    @property
    def record(self):
        """Return the reference record of the simulation."""
        if self.keep_record:
            return np.atleast_2d(self._record)
        else:
            logger.warning("Reference record has not been saved.")

    def reset(self, state=None):
        """Reset internal state."""
        self._iter = 0
        self._current_ref = self._reference_function(state, 0, False)
        if self.keep_record:
            self._record = []

    def update(self, state, time, finished=False):
        """Compute the state of the reference object."""
        ref = self._reference_function(state, time, finished)
        self._iter += 1

        if self.keep_record:
            ref_value = np.hstack((ref.pos, ref.vel, ref.euler, ref.omega_b))
            self._update_record(ref_value)

        self._current_ref = ref

        return ref

    @property
    def reference(self):
        """Return the reference."""
        return self._current_ref

    def _update_record(self, ref_value):
        self._record.append(ref_value)
        assert self._iter == len(self._record)

    def _reference_chooser(self, **kwargs):
        # CIRCLE
        if self._name == 'circle':
            if kwargs.get('speed', False):
                speed = kwargs['speed']
            else:
                speed = pi / 2.
            if kwargs.get('initial_angle', False):
                init_angle = kwargs['initial_angle']
            else:
                init_angle = 0.
            if kwargs.get('radius', False):
                radius = kwargs['radius']
            else:
                radius = 1.
            if kwargs.get('z_vel', False):
                z_vel = kwargs['z_vel']
            else:
                z_vel = 0.
            return partial(_circle_reference,
                           speed=speed,
                           init_angle=init_angle,
                           radius=radius,
                           z_vel=z_vel)
        # STATIONARY
        elif self._name == 'stationary':
            if kwargs.get('position', False):
                position = kwargs['position']
            else:
                position = [1., 0., 0.]
            return partial(_stationary_reference,
                           position=position)
        # OSCILLATE
        elif self._name == 'oscillate':
            if kwargs.get('x_velocity', False):
                x_vel = kwargs['x_velocity']
            else:
                x_vel = 0.5
            if kwargs.get('omega', False):
                omega = kwargs['omega']
            else:
                omega = 1.
            if kwargs.get('radius', False):
                radius = kwargs['radius']
            else:
                radius = 0.5
            return partial(_oscillate_reference,
                           x_vel=x_vel,
                           omega=omega,
                           radius=radius)


# private circle reference function
def _circle_reference(state,
                      time,
                      finished,
                      radius=None,
                      speed=None,
                      init_angle=None,
                      z_vel=None):
    ref = StateVector()
    angle = init_angle + speed / radius * time

    ref.pos = array([radius * cos(angle),
                     radius * sin(angle),
                     z_vel * time])
    ref.vel[:] = [-speed * sin(angle), speed * cos(angle), z_vel]
    ref.euler[2] = pi + np.arctan2(state.pos[1], state.pos[0])
    # reference.omega_b[2] = speed / radius
    return ref


# private stationary reference function
def _stationary_reference(state,
                          time,
                          finished,
                          position=None):
    ref = StateVector()
    ref.pos[0] = position[0]
    ref.pos[1] = position[1]
    ref.pos[2] = position[2]
    return ref


# private oscillation reference function
def _oscillate_reference(state,
                         time,
                         finished,
                         x_vel=None,
                         omega=None,
                         radius=None):
    ref = StateVector()
    angle = omega * time
    ref.pos[0] = x_vel * time
    ref.pos[1] = radius * sin(angle)
    ref.pos[2] = 0.
    ref.vel[0] = x_vel
    ref.vel[1] = radius * omega * cos(angle)
    ref.vel[2] = 0.
    return ref
