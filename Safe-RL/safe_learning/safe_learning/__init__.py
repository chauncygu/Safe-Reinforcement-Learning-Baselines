"""
The `safeopt` package implements tools for Safe Bayesian optimization.

Stability verification
----------------------

The :class:`Lyapunov` class provides the main point of entry for the stability
analysis. It can be used to compute the region of attraction and together with
:func:`get_safe_sample` sets up the safe sampling scheme.

.. autosummary::

   :template: template.rst
   :toctree:

   Lyapunov
   get_safe_sample
   smallest_boundary_value
   get_lyapunov_region


Approximate Dynamics Programming
--------------------------------

We use approximate dynamics programming to compute value functions.

.. autosummary::

   :template: template.rst
   :toctree:

   PolicyIteration


Functions
---------

These are generic function classes for convenience. They are all compatible
with :class:`Lyapunov` and :class:`PolicyIteration` and can be added,
multiplied, and stacked as needed.

.. autosummary::

   :template: template.rst
   :toctree:

   GridWorld
   FunctionStack
   Triangulation
   PiecewiseConstant
   LinearSystem
   QuadraticFunction
   Saturation
   NeuralNetwork
   GaussianProcess
   GPRCached
   sample_gp_function


Utilities
---------

These are utilities to make working with tensorflow more pleasant.

.. autosummary::

   :template: template.rst
   :toctree:

   utilities.combinations
   utilities.linearly_spaced_combinations
   utilities.lqr
   utilities.dlqr
   utilities.ellipse_bounds
   utilities.concatenate_inputs
   utilities.make_tf_fun
   utilities.with_scope
   utilities.use_parent_scope
   utilities.add_weight_constraint
   utilities.batchify
   utilities.get_storage
   utilities.set_storage
   utilities.unique_rows
   utilities.gradient_clipping

"""

from __future__ import absolute_import

# Add the configuration settings
from .configuration import Configuration
config = Configuration()
del Configuration

from .functions import *
from .lyapunov import *
from .reinforcement_learning import *
from . import utilities

try:
    from pytest import main as run_tests
except ImportError:
    def run_tests():
        """Run the test package."""
        raise ImportError('Testing requires the pytest package.')
