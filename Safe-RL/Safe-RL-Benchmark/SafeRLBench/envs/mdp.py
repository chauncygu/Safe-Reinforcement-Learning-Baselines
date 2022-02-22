"""Markov Decision Process Implementations."""

import numpy as np

from SafeRLBench import EnvironmentBase
from SafeRLBench.spaces import DiscreteSpace


class MDP(EnvironmentBase):
    """Discrete Markov Decision Process Environment.

    Attributes
    ----------
    transitions : array-like
        Array holding transition matrix for each action. The dimension of
        the state and action spaces will be deduced from this array.
    rewards : array-like
        Array holding the reward matrix for each action. It needs to comply
        with the dimensions deduced from the transitions array.
    action_space : DiscreteSpace object
        Action space as determined from the transitions array
    state_space : DiscreteSpace object
        State space as determined from the transitions array.
    init_state : int
        Initial state of the process. If None, it will be set to 0.
    state : int
        Current state of the system.
    """

    def __init__(self, transitions, rewards, horizon=100, init_state=None,
                 seed=None):
        """MDP initialization.

        Parameters
        ----------
        transitions : array-like
            Array holding transition matrix for each action. The dimension of
            the state and action spaces will be deduced from this array.
        rewards : array-like
            Array holding the reward matrix for each action. It needs to comply
            with the dimensions deduced from the transitions array.
        init_state : int
            Initial state of the process. If None, it will be set to 0.
        """
        self.horizon = horizon

        self.transitions = transitions
        self.rewards = rewards

        # determine state and action space
        self.action_space = DiscreteSpace(len(transitions))
        self.state_space = DiscreteSpace(len(transitions[0]))

        # if initial state is none, we will use 0 as an initial state
        if init_state is None:
            init_state = 0
        elif not self.state_space.contains(init_state):
            raise ValueError('Initial state (%d) is not a valid state.',
                             init_state)

        # setup current state and store the initial state for reset
        self.init_state = init_state
        self.state = init_state

        # generate random state
        self.random = np.random.RandomState()

        if seed is not None:
            self.seed = seed
        else:
            self._seed = None

    @property
    def seed(self):
        """Seed."""
        return self._seed

    @seed.setter
    def seed(self, v):
        self.random.seed(v)
        self._seed = v

    def _update(self, action):
        prev_state = self.state

        # choose next state
        self.state = self.random.choice(np.arange(self.state_space.dimension),
                                        p=self.transitions[action][self.state])
        # determine reward
        reward = self.rewards[action][prev_state][self.state]

        return action, self.state, reward

    def _reset(self):
        self.state = self.init_state


def _get_test_args():
    # private method that will generate arguments for mdp testing.
    transitions = [
        [[.1, .9, 0., 0., 0.],
         [.2, 0., .8, 0., 0.],
         [.3, 0., 0., .7, 0.],
         [.4, 0., 0., 0., .6],
         [.4, 0., 0., 0., .6]],
        [[1., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.]]
    ]

    rewards = [
        [[0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0.]],
        [[0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0.],
         [2., 0., 0., 0., 0.],
         [3., 0., 0., 0., 0.],
         [4., 0., 0., 0., 0.]],
    ]

    return [transitions, rewards, 100, None, 42]
