"""Discrete space implementation."""

from SafeRLBench import Space

import numpy as np


class DiscreteSpace(Space):
    """Discrete Space.

    Let d be the dimension of the space, then it will contain elements
    {0, 1, ... , dim-1}.

    Examples
    --------
    Create a `DiscreteSpace` with three states:
    >>> from SafeRLBench.spaces import DiscreteSpace
    >>> discrete_space = DiscreteSpace(3)
    """

    def __init__(self, dim):
        """Initialize `DiscreteSpace`.

        Parameters
        ----------
        dim : int
            Number of states.
        """
        assert dim > 0, ("If you need a discrete space without elements, you "
                         + "do not need this class.")
        self._dim = dim

    def contains(self, x):
        """Check if element is part of the space."""
        return (isinstance(x, int) and x >= 0 and x < self._dim)

    def sample(self):
        """Sample an element of the space."""
        return np.random.randint(self._dim)

    @property
    def dimension(self):
        """Return dimension of the space."""
        return self._dim

    def __repr__(self):
        return 'DiscreteSpace(dim=%d)' % self._dim
