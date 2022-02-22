=====================================================
Safe Reinforcement Learning with Stability Guarantees
=====================================================

.. image:: https://travis-ci.org/befelix/safe_learning.svg?branch=master
    :target: https://travis-ci.org/befelix/safe_learning
    :alt: Build status
.. image:: https://readthedocs.org/projects/safe-learning/badge/?version=latest
    :target: http://safe-learning.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

This code accompanies the paper [1]_ and implements the code for estimating the region of attraction for a policy and optimizing the policy subject to stability constraints. For the old numpy-based code to estimate the region of attraction in [2]_ see the `lyapunov-learning <https://github.com/befelix/lyapunov-learning>`_ repository. The code for learning Lyapunov functions from [3]_ can be found in the `examples <./examples>`_ folder.

.. [1] F. Berkenkamp, M. Turchetta, A. P. Schoellig, A. Krause,
  `Safe Model-based Reinforcement Learning with Stability Guarantees <http://arxiv.org/abs/1509.01066>`_
  in Proc. of the Conference on Neural Information Processing Systems (NIPS), 2017.

.. [2] F. Berkenkamp, R. Moriconi, A. P. Schoellig, A. Krause,
  `Safe Learning of Regions of Attraction in Uncertain, Nonlinear Systems with Gaussian Processes <http://arxiv.org/abs/1603.04915>`_
  in Proc. of the Conference on Decision and Control (CDC), 2016.

.. [3] S. M. Richards, F. Berkenkamp, A. Krause,
  `The Lyapunov Neural Network: Adaptive Stability Certification for Safe Learning of Dynamical Systems <https://arxiv.org/abs/1808.00924>`_. Conference on Robot Learning (CoRL), 2018.

Getting started
---------------

This library is tested based on both python 2.7 and 3.5, together with the following dependencies, since ``pip>=19`` does not support ``--process-dependency-links`` (see below)

::

  pip install pip==18.1
  pip install numpy==1.14.5


Based on this, you can install the library by cloning the repository and installing it with

``pip install . --process-dependency-links``

To run the tests with the bash script in ``scripts/test_code.sh``, you need to install additional dependencies with

``pip install ".[test]" --process-dependency-links``

The ``--process-dependency-links`` flag is needed to install ``gpflow==0.4.0``, which is not on pypi. You can skip it if that particular version of the library is already installed.

You can the find example jupyter notebooks and the experiments in the paper in the `examples <./examples>`_ folder.

