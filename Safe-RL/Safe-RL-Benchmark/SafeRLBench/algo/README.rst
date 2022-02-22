Description
-----------

The ``algo`` module contains algorithm implementations based on the
``AlgorithmBase`` class.
The objects should only be accessed through the interface functions defined
in the base class.

Overview
--------

=============== ===============
Algorithm       Policy
=============== ===============
A3C             NeuralNetwork
PolicyGradient  Any
Q-Learning      None
SafeOpt         Any
=============== ===============

Implementing an Algorithm
-------------------------

When implementing an algorithm a couple of things have to be considered.
``AlgorithmBase`` is an abstrace base class. It will require any subclass to
implement the private methods listed below. These will be invoked by the
public interface methods.

Any algorithm must be structured using four methods. First the ``optimize``,
which will control the optimization run, it is responsible for using the other
methods. The three tools ``optimize`` should use are the methods
``initialize``, ``step`` and ``is_finished``.

``initialize`` should be used to initialize the run and all the attributes and
parameters that need to be set up.
``optimize`` should compute one step of the optimization run.
``is_finished`` is supposed to return ``True`` when the optimization run is
finished.

Requirements
~~~~~~~~~~~~

================= =============================================================
Must implement
===============================================================================
_initialize       Initialize any attributes, objects needed.
_step             Execute one iteration of the algorithm.
_is_finished      Return ``True`` when done.
================= =============================================================

================= =============================================================
May implement
===============================================================================
_optimize(policy) Optimize the policy. Possibly no policy as in Q-learning.
================= =============================================================
