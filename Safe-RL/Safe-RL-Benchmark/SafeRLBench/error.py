"""Exceptions and error messages."""

import logging

logger = logging.getLogger(__name__)


class NotSupportedException(Exception):
    """Exception raised when requirements are not installed.

    Attributes
    ----------
    dep : Module
        The dependent module.
    dep_name : String
        Name of the dependency for a meaningful error message.
    """

    def __init__(self, dep, name='Some'):
        """Initialize NotSupportedException.

        Parameters
        ----------
        dep : Module
            The dependent module.
        dep_name : String
            Name of the dependency for a meaningful error message.
        """
        msg = name + " is not installed on this system."

        super(NotSupportedException, self).__init__(msg)

        self.dep = dep
        self.name = name


class MultipleCallsException(Exception):
    """Exception raised when a setup method is called multiple times."""

    pass


class IncompatibilityException(Exception):
    """Exception raised when any two parts are incompatible with each other.

    Attributes
    ----------
    obj1 : object
        Instance of the object calling the exception.
    obj2 : object
        Instance of the object being incompatible.
    """

    def __init__(self, obj1, obj2):
        """Initialize IncompatibilityException.

        Parameters
        ----------
        obj1 : object
            Instance of the object calling the exception.
        obj2 : object
            Instance of the object being incompatible.
        """
        msg = "%s is incompatible with %s." % (obj2.__name__,
                                               obj1.__name__)

        super(IncompatibilityException, self).__init__(msg)

        self.obj1 = obj1
        self.obj2 = obj2


def add_dependency(dep, dep_name='Some'):
    """Add dependency.

    Function, that will raise a `NotSupportedException` when `dep` is None.

    Parameters
    ----------
    dep : Module
        The dependent module.
    dep_name : String
        Name of the dependency for a meaningful error message.
    """
    if dep is None:
        raise NotSupportedException(dep, dep_name)
