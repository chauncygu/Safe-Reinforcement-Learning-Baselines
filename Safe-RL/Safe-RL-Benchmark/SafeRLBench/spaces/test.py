"""Tests for spaces module."""
from __future__ import absolute_import

from functools import partial
import inspect

from numpy import array
import SafeRLBench.spaces as spaces


"""Dictionary storing initialization arguments for classes."""
class_arguments = {
    spaces.BoundedSpace: [array([-1, -2]), array([1, 0])],
    spaces.RdSpace: [(3, 2)],
    spaces.DiscreteSpace: [5]
}


class TestSpaces(object):
    """Wrap spaces tests."""

    classes = []

    @classmethod
    def setUpClass(cls):
        """Initialize classes list."""
        for name, c in inspect.getmembers(spaces):
            if inspect.isclass(c):
                cls.classes.append(c)

    def exhaustive_tests(self):
        """Check: Spaces tests initial values for testing."""
        for c in self.classes:
            if c not in class_arguments:
                assert(False)

    def generate_tests(self):
        """Generate tests for spaces implementations."""
        for c in self.classes:
            if c in class_arguments:
                check = partial(self.check_contains)
                check.description = ('Test: ' + c.__name__.upper()
                                     + ': implementation.')
                yield check, c

    def check_contains(self, c):
        """Check if contains and element is implemented."""
        space = c(*class_arguments[c])
        try:
            x = space.sample()
            b = space.contains(x)
        except NotImplementedError:
            assert(False)
        assert(b)
