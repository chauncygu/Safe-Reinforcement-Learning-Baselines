from __future__ import absolute_import

import logging

from .configuration import SRBConfig

# Initialize configuration
config = SRBConfig(logging.getLogger(__name__))

from .monitor import AlgoMonitor, EnvMonitor
from .base import EnvironmentBase, Space, AlgorithmBase, Policy, ProbPolicy
from .bench import Bench, BenchConfig
from . import algo
from . import envs
from . import policy
from . import spaces
from . import error
from . import measure

# Add things to all
__all__ = ['EnvironmentBase',
           'Space',
           'AlgorithmBase',
           'Policy',
           'ProbPolicy',
           'AlgoMonitor',
           'EnvMonitor',
           'SRBConfig',
           'Bench',
           'BenchConfig',
           'envs',
           'algo',
           'policy',
           'spaces',
           'measure',
           'error']


# Import test after __all__ (no documentation)
# from numpy.testing import Tester
# test = Tester().test
