"""R^d with any shape."""
import numpy as np
from SafeRLBench import Space


class RdSpace(Space):
    """R^d Vectorspace."""

    def __init__(self, shape):
        """Initialize with shape."""
        self.shape = shape
        self._dim = None

    def contains(self, x):
        """Check if element is contained."""
        return isinstance(x, np.ndarray) and x.shape == self.shape

    def sample(self):
        """Return arbitrary element."""
        return np.ones(self.shape)

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
        return 'RdSpace(shape=%s)' % str(self.shape)
