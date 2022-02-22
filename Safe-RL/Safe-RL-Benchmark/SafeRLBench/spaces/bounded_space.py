"""Bounded subspace of R^n."""
import numpy as np
from SafeRLBench import Space

from numpy.random import rand


class BoundedSpace(Space):
    """Bounded subspace of R^n.

    Attributes
    ----------
    lower : array-like
        Lower bound
    upper : array-like
        Upper bound

    Examples
    --------
    The `BoundedSpace` class can be instatiated in two ways. If you have
    individual bounds for each dimension, then you can directly pass the
    `lower` or `upper` bound as an array-like.

    >>> space = BoundedSpace(np.array([-1, -2]), np.array([1, 0]))

    In this case the shape argument will be ignored. If you want to create a
    however shaped box, where all the bounds are the same, then, you may pass
    a lower and an upper bound as a scalar and make sure that you specify the
    shape.

    >>> space = BoundedSpace(-1, 1, shape=(2,))
    """

    def __init__(self, lower, upper, shape=None):
        """Initialize BoundedSpace.

        Parameters
        ----------
        lower : array-like
            Lower bound of the space. Either an array or an integer.
            Must agree with the input of the upper bound.
        upper : array-like
            Upper bound of the space. Either an array or an integer. Must
            agree with the input of the lower bound.
        shape : integer
            Shape of the bounds. Input will be ignored, if the bounds are non
            scalar, if they are scalar, it must be set.
        """
        if (np.isscalar(lower) and np.isscalar(upper)):
            assert shape is not None, "Shape must be set, if bounds are scalar"
            self.lower = np.zeros(shape) + lower
            self.upper = np.zeros(shape) + upper
        else:
            self.lower = np.array(lower)
            self.upper = np.array(upper)
            assert self.lower.shape == self.upper.shape, "Shapes do not agree."

        self._dim = None

    def contains(self, x):
        """Check if element is contained."""
        return (x.shape == self.lower.shape
                and (x >= self.lower).all()
                and (x <= self.upper).all())

    def sample(self):
        """Return element."""
        element = rand(*self.shape) * (self.upper - self.lower) + self.lower
        return element

    @property
    def shape(self):
        """Return element shape."""
        return self.lower.shape

    @property
    def dimension(self):
        """Return dimension of the space."""
        if self._dim is None:
            d = 1
            for i in range(len(self.shape)):
                d *= self.shape[i]
            self._dim = d
        return self._dim

    def __repr__(self):
        return 'BoundedSpace(lower=%s, upper=%s)' % (str(self.lower),
                                                     str(self.upper))
