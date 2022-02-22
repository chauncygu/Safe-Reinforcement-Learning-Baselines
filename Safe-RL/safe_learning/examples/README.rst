Example notebooks for the library
=================================

Introductions
-------------
- `1d_region_of_attraction_estimate.ipynb <./1d_region_of_attraction_estimate.ipynb>`_ shows how to estimate and learn the region of attraction for a fixed policy.
- `basic_dynamic_programming.ipynb <./basic_dynamic_programming.ipynb>`_ does basic dynamic programming with piecewise linear function approximators for the mountain car example.
- `reinforcement_learning_pendulum.ipynb <./reinforcement_learning_pendulum.ipynb>`_ does approximate policy iteration in an actor-critic framework with neural networks for the inverted pendulum.
- `reinforcement_learning_cartpole.ipynb <./reinforcement_learning_cartpole.ipynb>`_ does the same as above for the cart-pole (i.e., the inverted pendulum on a cart).

Experiments
-----------
- `1d_example.ipynb <./1d_example.ipynb>`_ contains a 1D example including plots of the sets.
- `inverted_pendulum.ipynb <./inverted_pendulum.ipynb>`_ contains a full neural network example with an inverted pendulum.
- `adaptive_safety_verification.ipynb <./adaptive_safety_verification.ipynb>`_ investigates the benefits of an adaptive discretization in identifying safe sets for the inverted pendulum.
- `lyapunov_function_learning.ipynb <./lyapunov_function_learning.ipynb>`_ demonstrates how a parameterized Lyapunov candidate for the inverted pendulum can be trained with the machine learning approach in [1]_.

.. [1] S. M. Richards, F. Berkenkamp, A. Krause,
  `The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems <https://arxiv.org/abs/1808.00924>`_. Conference on Robot Learning (CoRL), 2018.
