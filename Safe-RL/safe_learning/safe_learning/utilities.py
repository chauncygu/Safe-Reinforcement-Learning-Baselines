"""
Utilities for plotting, function definitions, and GPs.

This file defines utilities needed for the experiments, such as creating
parameter grids, computing LQR controllers, Lyapunov functions, sample
functions of Gaussian processes, and plotting ellipses.

Author: Felix Berkenkamp, Learning & Adaptive Systems Group, ETH Zurich
        (GitHub: befelix)
"""

from __future__ import absolute_import, division, print_function

import itertools
import inspect
from functools import wraps, partial

import numpy as np
import scipy.interpolate
import scipy.linalg
import tensorflow as tf
from future.builtins import zip, range
from future.backports import OrderedDict

from safe_learning import config

__all__ = ['combinations', 'linearly_spaced_combinations', 'lqr', 'dlqr',
           'ellipse_bounds', 'concatenate_inputs', 'make_tf_fun',
           'with_scope', 'use_parent_scope', 'add_weight_constraint',
           'batchify', 'get_storage', 'set_storage', 'unique_rows',
           'gradient_clipping']


_STORAGE = {}


def make_tf_fun(return_type, gradient=None, stateful=True):
    """Convert a python function to a tensorflow function.

    Parameters
    ----------
    return_type : list
        A list of tensorflow return types. Needs to match with the gradient.
    gradient : callable, optional
        A function that provides the gradient. It takes `op` and one gradient
        per output of the function as inputs and returns one gradient for each
        input of the function. If stateful is `False` then tensorflow does not
        seem to compute gradients at all.

    Returns
    -------
    A tensorflow function with gradients registered.
    """
    def wrap(function):
        """Create a new function."""
        # Function name with stipped underscore (not allowed by tensorflow)
        name = function.__name__.lstrip('_')

        # Without gradients we can take the short route here
        if gradient is None:
            @wraps(function)
            def wrapped_function(self, *args, **kwargs):
                method = partial(function, self, **kwargs)
                return tf.py_func(method, args, return_type,
                                  stateful=stateful, name=name)

            return wrapped_function

        # Name for the gradient operation
        grad_name = name + '_gradient'

        @wraps(function)
        def wrapped_function(self, *args):
            # Overwrite the gradient
            graph = tf.get_default_graph()

            # Make sure the name we specify is unique
            unique_grad_name = graph.unique_name(grad_name)

            # Register the new gradient method with tensorflow
            tf.RegisterGradient(unique_grad_name)(gradient)

            # Remove self: Tensorflow does not allow for non-tensor inputs
            method = partial(function, self)

            with graph.gradient_override_map({"PyFunc": unique_grad_name}):
                return tf.py_func(method, args, return_type,
                                  stateful=stateful, name=name)

        return wrapped_function
    return wrap


def with_scope(name):
    """Set the tensorflow scope for the function.

    Parameters
    ----------
    name : string, optional

    Returns
    -------
    The tensorflow function with scope name.
    """
    def wrap(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            with tf.name_scope(name):
                return function(*args, **kwargs)
        return wrapped_function
    return wrap


def use_parent_scope(function):
    """Use the parent scope for tensorflow."""
    @wraps(function)
    def wrapped_function(self, *args, **kwargs):
        with tf.variable_scope(self.scope_name):
            return function(self, *args, **kwargs)
    return wrapped_function


def concatenate_inputs(start=0):
    """Concatenate the numpy array inputs to the functions.

    Parameters
    ----------
    start : int, optional
        The attribute number at which to start concatenating.
    """
    def wrap(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            """Concatenate the input arguments."""
            nargs = len(args) - start
            # Check for tensorflow objects
            tf_objects = (tf.Tensor, tf.Variable)
            if any(isinstance(arg, tf_objects) for arg in args[start:]):
                # reduce number of function calls in graph
                if nargs == 1:
                    return function(*args, **kwargs)
                # concatenate extra arguments
                args = args[:start] + (tf.concat(args[start:], axis=1),)
                return function(*args, **kwargs)
            else:
                # Map to 2D objects
                to_concatenate = map(np.atleast_2d, args[start:])

                if nargs == 1:
                    concatenated = tuple(to_concatenate)
                else:
                    concatenated = (np.hstack(to_concatenate),)

                args = args[:start] + concatenated
                return function(*args, **kwargs)

        return wrapped_function

    return wrap


def add_weight_constraint(optimization, var_list, bound_list):
    """Add weight constraints to an optimization step.

    Parameters
    ----------
    optimization : tf.Tensor
        The optimization routine that updates the parameters.
    var_list : list
        A list of variables that should be bounded.
    bound_list : list
        A list of bounds (lower, upper) for each variable in var_list.

    Returns
    -------
    assign_operations : list
        A list of assign operations that correspond to one step of the
        constrained optimization.
    """
    with tf.control_dependencies([optimization]):
        new_list = []
        for var, bound in zip(var_list, bound_list):
            clipped_var = tf.clip_by_value(var, bound[0], bound[1])
            assign = tf.assign(var, clipped_var)
            new_list.append(assign)
    return new_list


def gradient_clipping(optimizer, loss, var_list, limits):
    """Clip the gradients for the optimization problem.

    Parameters
    ----------
    optimizer : instance of tensorflow optimizer
    loss : tf.Tensor
        The loss that we want to optimize.
    var_list : tuple
        A list of variables for which we want to compute gradients.
    limits : tuple
        A list of tuples with lower/upper bounds for each variable.

    Returns
    -------
    opt : tf.Tensor
        One optimization step with clipped gradients.

    Examples
    --------
    >>> from safe_learning.utilities import gradient_clipping
    >>> var = tf.Variable(1.)
    >>> loss = tf.square(var - 1.)
    >>> optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    >>> opt_loss = gradient_clipping(optimizer, loss, [var], [(-1, 1)])
    """
    gradients = optimizer.compute_gradients(loss, var_list=var_list)

    clipped_gradients = [(tf.clip_by_value(grad, low, up), var)
                         for (grad, var), (low, up) in zip(gradients, limits)]

    # Return optimization step
    return optimizer.apply_gradients(clipped_gradients)


def batchify(arrays, batch_size):
    """Yield the arrays in batches and in order.

    The last batch might be smaller than batch_size.

    Parameters
    ----------
    arrays : list of ndarray
        The arrays that we want to convert to batches.
    batch_size : int
        The size of each individual batch.
    """
    if not isinstance(arrays, (list, tuple)):
        arrays = (arrays,)

    # Iterate over array in batches
    for i, i_next in zip(itertools.count(start=0, step=batch_size),
                         itertools.count(start=batch_size, step=batch_size)):

        batches = [array[i:i_next] for array in arrays]

        # Break if there are no points left
        if batches[0].size:
            yield i, batches
        else:
            break


def combinations(arrays):
    """Return a single array with combinations of parameters.

    Parameters
    ----------
    arrays : list of np.array

    Returns
    -------
    array : np.array
        An array that contains all combinations of the input arrays
    """
    return np.array(np.meshgrid(*arrays)).T.reshape(-1, len(arrays))


def linearly_spaced_combinations(bounds, num_samples):
    """
    Return 2-D array with all linearly spaced combinations with the bounds.

    Parameters
    ----------
    bounds : sequence of tuples
        The bounds for the variables, [(x1_min, x1_max), (x2_min, x2_max), ...]
    num_samples : integer or array_likem
        Number of samples to use for every dimension. Can be a constant if
        the same number should be used for all, or an array to fine-tune
        precision. Total number of data points is num_samples ** len(bounds).

    Returns
    -------
    combinations : 2-d array
        A 2-d arrray. If d = len(bounds) and l = prod(num_samples) then it
        is of size l x d, that is, every row contains one combination of
        inputs.
    """
    bounds = np.atleast_2d(bounds)
    num_vars = len(bounds)
    num_samples = np.broadcast_to(num_samples, num_vars)

    # Create linearly spaced test inputs
    inputs = [np.linspace(b[0], b[1], n) for b, n in zip(bounds,
                                                         num_samples)]

    # Convert to 2-D array
    return combinations(inputs)


def lqr(a, b, q, r):
    """Compute the continuous time LQR-controller.

    The optimal control input is `u = -k.dot(x)`.

    Parameters
    ----------
    a : np.array
    b : np.array
    q : np.array
    r : np.array

    Returns
    -------
    k : np.array
        Controller matrix
    p : np.array
        Cost to go matrix
    """
    a, b, q, r = map(np.atleast_2d, (a, b, q, r))
    p = scipy.linalg.solve_continuous_are(a, b, q, r)

    # LQR gain
    k = np.linalg.solve(r, b.T.dot(p))

    return k, p


def dlqr(a, b, q, r):
    """Compute the discrete-time LQR controller.

    The optimal control input is `u = -k.dot(x)`.

    Parameters
    ----------
    a : np.array
    b : np.array
    q : np.array
    r : np.array

    Returns
    -------
    k : np.array
        Controller matrix
    p : np.array
        Cost to go matrix
    """
    a, b, q, r = map(np.atleast_2d, (a, b, q, r))
    p = scipy.linalg.solve_discrete_are(a, b, q, r)

    # LQR gain
    # k = (b.T * p * b + r)^-1 * (b.T * p * a)
    bp = b.T.dot(p)
    tmp1 = bp.dot(b)
    tmp1 += r
    tmp2 = bp.dot(a)
    k = np.linalg.solve(tmp1, tmp2)

    return k, p


def ellipse_bounds(P, level, n=100):
    """Compute the bounds of a 2D ellipse.

    The levelset of the ellipsoid is given by
    level = x' P x. Given the coordinates of the first
    dimension, this function computes the corresponding
    lower and upper values of the second dimension and
    removes any values of x0 that are outside of the ellipse.

    Parameters
    ----------
    P : np.array
        The matrix of the ellipsoid
    level : float
        The value of the levelset
    n : int
        Number of data points

    Returns
    -------
    x : np.array
        1D array of x positions of the ellipse
    yu : np.array
        The upper bound of the ellipse
    yl : np.array
        The lower bound of the ellipse

    Notes
    -----
    This can be used as
    ```plt.fill_between(*ellipse_bounds(P, level))```
    """
    # Round up to multiple of 2
    n += n % 2

    # Principal axes of ellipsoid
    eigval, eigvec = np.linalg.eig(P)
    eigvec *= np.sqrt(level / eigval)

    # set zero angle at maximum x
    angle = np.linspace(0, 2 * np.pi, n)[:, None]
    angle += np.arctan(eigvec[0, 1] / eigvec[0, 0])

    # Compute positions
    pos = np.cos(angle) * eigvec[:, 0] + np.sin(angle) * eigvec[:, 1]
    n /= 2

    # Return x-position (symmetric) and upper/lower bounds
    return pos[:n, 0], pos[:n, 1], pos[:n - 1:-1, 1]


def get_storage(dictionary, index=None):
    """Get a unique storage point within a class method.

    Parameters
    ----------
    dictionary : dict
        A dictionary used for storage.
    index : hashable
        An index under which to store the element. Needs to be hashable.
        This is useful for functions which might be accessed with multiple
        different arguments.

    Returns
    -------
    storage : OrderedDict
        The storage object. Is None if no storage exists. Otherwise it
        returns the OrderedDict that was previously put in the storage.
    """
    # Use function name as storage name
    frame = inspect.currentframe()
    storage_name = inspect.getframeinfo(frame.f_back).function

    storage = dictionary.get(storage_name)

    if index is None:
        return storage
    elif storage is not None:
        # Return directly the indexed object
        try:
            return storage[index]
        except KeyError:
            pass


def set_storage(dictionary, name_value, index=None):
    """Set the storage point within a class method.

    Parameters
    ----------
    dictionary : dict
    name_value : tuple
        A list of tuples, where each tuple contains a string with the name
        of the storage object and the corresponding value that is to be put
        in storage. These are stored as OrderedDicts.
    index : hashable
        An index under which to store the element. Needs to be hashable.
        This is useful for functions which might be accessed with multiple
        different arguements.
    """
    # Use function name as storage name
    frame = inspect.currentframe()
    storage_name = inspect.getframeinfo(frame.f_back).function

    storage = OrderedDict(name_value)
    if index is None:
        dictionary[storage_name] = storage
    else:
        # Make sure the storage is initialized
        if storage_name not in dictionary:
            dictionary[storage_name] = {}
        # Set the indexed storage
        dictionary[storage_name][index] = storage


def get_feed_dict(graph):
    """Return the global feed_dict used for this graph.

    Parameters
    ----------
    graph : tf.Graph

    Returns
    -------
    feed_dict : dict
        The feed_dict for this graph.
    """
    try:
        # Just return the feed_dict
        return graph.feed_dict_sl
    except AttributeError:
        # Create a new feed_dict for this graph
        graph.feed_dict_sl = {}
        return graph.feed_dict_sl


def unique_rows(array):
    """Return the unique rows of the array.

    Parameters
    ----------
    array : ndarray
        A 2D numpy array.

    Returns
    -------
    unique_array : ndarray
        A 2D numpy array that contains all the unique rows of array.
    """
    array = np.ascontiguousarray(array)
    # Combine all the rows into a single element of the flexible void datatype
    dtype = np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    combined_array = array.view(dtype=dtype)
    # Get all the unique rows of the combined array
    _, idx = np.unique(combined_array, return_index=True)

    return array[idx]


def compute_trajectory(dynamics, policy, initial_state, num_steps):
    """Compute a state trajectory given dynamics and a policy.

    Parameters
    ----------
    dynamics : callable
        A function that takes the current state and action as input and returns
        the next state.
    policy : callable
        A function that takes the current state as input and returns the
        action.
    initial_state : Tensor or ndarray
        The initial state at which to start simulating.
    num_steps : int
        The number of steps for which to simulate the system.

    Returns
    -------
    states : ndarray
        A (num_steps x state_dim) array with one state on each row.
    actions : ndarray
        A (num_steps x action_dim) array with the corresponding action on each
        row.
    """
    initial_state = np.atleast_2d(initial_state)
    state_dim = initial_state.shape[1]

    # Get storage (indexed by dynamics and policy)
    index = (dynamics, policy)
    storage = get_storage(_STORAGE, index=index)

    if storage is None:
        # Compute next state under the policy
        tf_state = tf.placeholder(config.dtype, [1, state_dim])
        tf_action = policy(tf_state)
        tf_next_state = dynamics(tf_state, tf_action)

        storage = [('tf_state', tf_state),
                   ('tf_action', tf_action),
                   ('tf_next_state', tf_next_state)]

        set_storage(_STORAGE, storage, index=index)
    else:
        tf_state, tf_action, tf_next_state = storage.values()

    # Initialize
    dtype = config.np_dtype
    states = np.empty((num_steps, state_dim), dtype=dtype)
    actions = np.empty((num_steps - 1, policy.output_dim), dtype=dtype)

    states[0, :] = initial_state

    # Get the feed dict
    session = tf.get_default_session()
    feed_dict = get_feed_dict(session.graph)

    next_data = [tf_next_state, tf_action]

    # Run simulation
    for i in range(num_steps - 1):
        feed_dict[tf_state] = states[[i], :]
        states[i + 1, :], actions[i, :] = session.run(next_data,
                                                      feed_dict=feed_dict)

    return states, actions
