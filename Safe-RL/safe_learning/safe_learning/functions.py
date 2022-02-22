"""An efficient implementation of Delaunay triangulation on regular grids."""

from __future__ import absolute_import, print_function, division

from types import ModuleType
from itertools import product as cartesian
from functools import partial

from future.builtins import zip, range
from scipy import spatial, sparse, linalg
import tensorflow as tf
import numpy as np
try:
    import gpflow
except ImportError as exception:
    gpflow = exception

from .utilities import (concatenate_inputs, make_tf_fun, with_scope,
                        use_parent_scope, get_feed_dict)
from safe_learning import config

__all__ = ['DeterministicFunction', '_Triangulation', 'Triangulation',
           'PiecewiseConstant', 'GridWorld', 'UncertainFunction',
           'FunctionStack', 'QuadraticFunction', 'GaussianProcess',
           'GPRCached', 'sample_gp_function', 'LinearSystem', 'Saturation',
           'NeuralNetwork']

_EPS = np.finfo(config.np_dtype).eps


class Function(object):
    """TensorFlow function baseclass.

    It makes sure that variables are reused if the function is called
    multiple times with a TensorFlow template.
    """

    def __init__(self, name='function'):
        super(Function, self).__init__()
        self.feed_dict = get_feed_dict(tf.get_default_graph())

        # Reserve the TensorFlow scope immediately to avoid problems with
        # Function instances with the same `name`
        with tf.variable_scope(name) as scope:
            self._scope = scope

        # Use `original_name_scope` explicitly in case `scope_name` method is
        # overridden in a child class
        self._template = tf.make_template(self._scope.original_name_scope,
                                          self.build_evaluation,
                                          create_scope_now_=True)

    @property
    def scope_name(self):
        return self._scope.original_name_scope

    @property
    def parameters(self):
        """Return the variables within the current scope."""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=self.scope_name)

    @use_parent_scope
    def __call__(self, *args, **kwargs):
        """Evaluate the function using the template to ensure variable sharing.

        Parameters
        ----------
        args : list
            The input arguments to the function.
        kwargs : dict, optional
            The keyword arguments to the function.

        Returns
        -------
        outputs : list
            The output arguments of the function as given by evaluate.

        """
        with tf.name_scope('evaluate'):
            outputs = self._template(*args, **kwargs)
        return outputs

    def build_evaluation(self, *args, **kwargs):
        """Build the function evaluation tree.

        Parameters
        ----------
        args : list
        kwargs : dict, optional

        Returns
        -------
        outputs : list

        """
        raise NotImplementedError('This function has to be implemented by the'
                                  'child class.')

    def copy_parameters(self, other_instance):
        """Copy over the parameters of another instance."""
        assign_ops = []
        for param, other_param in zip(self.parameters,
                                      other_instance.parameters):
            op = tf.assign(param, other_param, validate_shape=True,
                           name='copy_op')
            assign_ops.append(op)

        sess = tf.get_default_session()
        sess.run(assign_ops)

    def __add__(self, other):
        """Add this function to another."""
        return AddedFunction(self, other)

    def __mul__(self, other):
        """Multiply this function with another."""
        return MultipliedFunction(self, other)

    def __neg__(self):
        """Negate the function."""
        return MultipliedFunction(self, -1)


class AddedFunction(Function):
    """A class for adding two individual functions.

    Parameters
    ----------
    fun1 : instance of Function or scalar
    fun2 : instance of Function or scalar

    """

    def __init__(self, fun1, fun2):
        """Initialization, see `AddedFunction`."""
        super(AddedFunction, self).__init__()

        if not isinstance(fun1, Function):
            fun1 = ConstantFunction(fun1)
        if not isinstance(fun2, Function):
            fun2 = ConstantFunction(fun2)

        self.fun1 = fun1
        self.fun2 = fun2

    @property
    def parameters(self):
        """Return the parameters."""
        return self.fun1.parameters + self.fun2.parameters

    def copy_parameters(self, other_instance):
        """Return a copy of the function (new tf variables with same values."""
        return AddedFunction(self.fun1.copy_parameters(other_instance.fun1),
                             self.fun2.copy_parameters(other_instance.fun1))

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Evaluate the function."""
        return self.fun1(points) + self.fun2(points)


class MultipliedFunction(Function):
    """A class for pointwise multiplying two individual functions.

    Parameters
    ----------
    fun1 : instance of Function or scalar
    fun2 : instance of Function or scalar

    """

    def __init__(self, fun1, fun2):
        """Initialization, see `AddedFunction`."""
        super(MultipliedFunction, self).__init__()

        if not isinstance(fun1, Function):
            fun1 = ConstantFunction(fun1)
        if not isinstance(fun2, Function):
            fun2 = ConstantFunction(fun2)

        self.fun1 = fun1
        self.fun2 = fun2

    @property
    def parameters(self):
        """Return the parameters."""
        return self.fun1.parameters + self.fun2.parameters

    def copy_parameters(self, other_instance):
        """Return a copy of the function (copies parameters)."""
        copied_fun1 = self.fun1.copy_parameters(other_instance.fun1)
        copied_fun2 = self.fun2.copy_parameters(other_instance.fun2)
        return MultipliedFunction(copied_fun1, copied_fun2)

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Evaluate the function."""
        return self.fun1(points) * self.fun2(points)


class UncertainFunction(Function):
    """Base class for function approximators."""

    def __init__(self, **kwargs):
        """Initialization, see `UncertainFunction`."""
        super(UncertainFunction, self).__init__(**kwargs)

    def to_mean_function(self):
        """Turn the uncertain function into a deterministic 'mean' function."""
        def _only_first_output(function):
            """Remove all but the first output of a function.

            Parameters
            ----------
            function : callable

            Returns
            -------
            function : callable
                The modified function.

            """
            def new_function(*points):
                return function(*points)[0]
            return new_function

        new_evaluate = _only_first_output(self.build_evaluation)

        return new_evaluate


class DeterministicFunction(Function):
    """Base class for function approximators."""

    def __init__(self, **kwargs):
        """Initialization, see `Function` for details."""
        super(DeterministicFunction, self).__init__(**kwargs)


class ConstantFunction(DeterministicFunction):
    """A function with a constant value."""

    def __init__(self, constant, name='constant_function'):
        """Initialize, see `ConstantFunction`."""
        super(ConstantFunction, self).__init__(name=name)
        self.constant = constant

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        return self.constant


class FunctionStack(UncertainFunction):
    """A combination of multiple 1d (uncertain) functions for each dim.

    Parameters
    ----------
    functions : list
        The functions. There should be one for each dimension of the output.

    """

    def __init__(self, functions, name='function_stack'):
        """Initialization, see `FunctionStack`."""
        super(FunctionStack, self).__init__(name=name)
        self.functions = functions
        self.num_fun = len(self.functions)

        self.input_dim = self.functions[0].input_dim
        self.output_dim = sum(fun.output_dim for fun in self.functions)

    @property
    def parameters(self):
        """Return the parameters."""
        return sum(fun.parameters for fun in self.functions)

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Evaluation, see `UncertainFunction.evaluate`."""
        means = []
        errors = []
        for fun in self.functions:
            mean, error = fun(points)
            means.append(mean)
            errors.append(error)

        mean = tf.concat(means, axis=1, name='stacked_mean')
        error = tf.concat(errors, axis=1, name='stacked_error')

        return mean, error

    def add_data_point(self, x, y):
        """Add data points to the GP model and update cholesky.

        Parameters
        ----------
        x : ndarray
            A 2d array with the new states to add to the GP model. Each new
            state is on a new row.
        y : ndarray
            A 2d array with the new measurements to add to the GP model.
            Each measurements is on a new row.

        """
        for fun, yi in zip(self.functions, y.squeeze()):
            fun.add_data_point(x, yi)


class Saturation(DeterministicFunction):
    """Saturate the output of a `DeterministicFunction`.

    Parameters
    ----------
    fun : instance of `DeterministicFunction`.
    lower : float or arraylike
        Lower bound. Passed to `tf.clip_by_value`.
    upper : float or arraylike
        Upper bound. Passed to `tf.clip_by_value`.

    """

    def __init__(self, fun, lower, upper, name='saturation'):
        """Initialization. See `Saturation`."""
        super(Saturation, self).__init__(name=name)
        self.fun = fun
        self.lower = lower
        self.upper = upper

        self.input_dim = self.fun.input_dim
        self.output_dim = self.fun.output_dim

        # Copy over attributes and functions from fun
        for par in dir(self.fun):
            if par.startswith('__') or hasattr(self, par):
                continue
            setattr(self, par, eval('self.fun.' + par))

    @property
    def scope_name(self):
        """Return the scope name of the wrapped function."""
        return self.fun.scope_name

    def copy_parameters(self, other_instance):
        """Return a copy of the function (copies parameters)."""
        return Saturation(self.fun.copy_parameters(other_instance.fun),
                          self.lower, self.upper)

    def build_evaluation(self, points):
        """Evaluation, see `DeterministicFunction.evaluate`."""
        res = self.fun.build_evaluation(points)

        # Broadcasting in tf.clip_by_value not available in TensorFlow >= 1.6.0
        return tf.minimum(tf.maximum(res, self.lower), self.upper)


class GPRCached(gpflow.gpr.GPR):
    """gpflow.gpr.GPR class that stores cholesky decomposition for efficiency.

    Parameters
    ----------
    x : ndarray
        A 2d array with states to initialize the GP model. Each state is on
        a row.
    y : ndarray
        A 2d array with measurements to initialize the GP model. Each
        measurement is on a row.
    scale : float, optional
        An internal scaling factor used during GP prediction for improved
        numerical stability.

    """

    def __init__(self, x, y, kern, mean_function=gpflow.mean_functions.Zero(),
                 scale=1., name='GPRCached'):
        """Initialize GP and cholesky decomposition."""
        # Make sure gpflow is imported
        if not isinstance(gpflow, ModuleType):
            raise gpflow

        # self.scope_name = scope.original_name_scope
        gpflow.gpr.GPR.__init__(self, x, y, kern, mean_function, name)

        # Create new dataholders for the cached data
        # TODO zero-dim dataholders cause strange allocator errors in
        # tensorflow with MKL
        dtype = config.np_dtype
        self.cholesky = gpflow.param.DataHolder(np.empty((0, 0), dtype=dtype),
                                                on_shape_change='pass')
        self.alpha = gpflow.param.DataHolder(np.empty((0, 0), dtype=dtype),
                                             on_shape_change='pass')
        self._scale = scale
        self.update_cache()

    @with_scope('compute_cache')
    @gpflow.param.AutoFlow()
    def _compute_cache(self):
        """Compute cache."""
        # Scaled kernel
        identity = tf.eye(tf.shape(self.X)[0], dtype=config.dtype)
        kernel = self.kern.K(self.X) + identity * self.likelihood.variance
        kernel *= (self._scale ** 2)

        # Scaled target
        target = self._scale * (self.Y - self.mean_function(self.X))

        # Cholesky decomposition
        cholesky = tf.cholesky(kernel, name='gp_cholesky')
        alpha = tf.matrix_triangular_solve(cholesky, target, name='gp_alpha')

        return cholesky, alpha

    def update_cache(self):
        """Update the cache after adding data points."""
        self.cholesky, self.alpha = self._compute_cache()

    @with_scope('build_predict')
    def build_predict(self, Xnew, full_cov=False):
        """Predict mean and variance of the GP at locations in Xnew.

        Parameters
        ----------
        Xnew : ndarray
            The points at which to evaluate the function. One row for each
            data points.
        full_cov : bool
            if False returns only the diagonal of the covariance matrix

        Returns
        -------
        mean : ndarray
            The expected function values at the points.
        error_bounds : ndarray
            Diagonal of the covariance matrix (or full matrix).

        """
        # Scaled kernel and mean
        Kx = (self._scale ** 2) * self.kern.K(self.X, Xnew)
        mx = self._scale * self.mean_function(Xnew)

        a = tf.matrix_triangular_solve(self.cholesky, Kx, lower=True)
        fmean = (tf.matmul(a, self.alpha, transpose_a=True) + mx)

        if full_cov:
            Knew = (self._scale ** 2) * self.kern.K(Xnew)
            fvar = Knew - tf.matmul(a, a, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            Knew = (self._scale ** 2) * self.kern.Kdiag(Xnew)
            fvar = Knew - tf.reduce_sum(tf.square(a), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])

        # Apply inverse-scaling to mean and variance before returning
        fmean /= self._scale
        fvar /= (self._scale ** 2)

        return fmean, fvar


class GaussianProcess(UncertainFunction):
    """A GaussianProcess model based on gpflow.

    Parameters
    ----------
    gaussian_process : instance of gpflow.models.GPModel
        The Gaussian process model.
    beta : float
        The scaling factor for the standard deviation to create
        confidence intervals.

    Notes
    -----
    The evaluate and gradient functions can be called with multiple
    arguments, in which case they are concatenated before being
    passed to the GP.

    """

    def __init__(self, gaussian_process, beta=2., name='gaussian_process'):
        """Initialization."""
        super(GaussianProcess, self).__init__(name=name)

        with tf.variable_scope(self.scope_name):
            self.n_dim = gaussian_process.X.shape[-1]
            self.gaussian_process = gaussian_process
            self.beta = float(beta)

            self.input_dim = gaussian_process.X.shape[1]
            self.output_dim = gaussian_process.Y.shape[1]

            self.hyperparameters = [tf.placeholder(config.dtype, [None])]
            self.gaussian_process.make_tf_array(self.hyperparameters[0])

            self.update_feed_dict()

    @property
    def X(self):
        """Input location of observed data. One observation per row."""
        return self.gaussian_process.X.value

    @property
    def Y(self):
        """Observed output. One observation per row."""
        return self.gaussian_process.Y.value

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Evaluate the model, but return tensorflow tensors."""
        # Build normal prediction
        with self.gaussian_process.tf_mode():
            mean, var = self.gaussian_process.build_predict(points)
        # Construct confidence intervals
        std = self.beta * tf.sqrt(var, name='standard_deviation')
        return mean, std

    def update_feed_dict(self):
        """Update the feed dictionary for tensorflow."""
        gp = self.gaussian_process
        feed_dict = self.feed_dict

        gp.update_feed_dict(gp.get_feed_dict_keys(), feed_dict)
        feed_dict[self.hyperparameters[0]] = gp.get_free_state()

    @use_parent_scope
    @with_scope('add_data_point')
    def add_data_point(self, x, y):
        """Add data points to the GP model and update cholesky.

        Parameters
        ----------
        x : ndarray
            A 2d array with the new states to add to the GP model. Each new
            state is on a new row.
        y : ndarray
            A 2d array with the new measurements to add to the GP model.
            Each measurements is on a new row.

        """
        gp = self.gaussian_process
        gp.X = np.vstack((self.X, np.atleast_2d(x)))
        gp.Y = np.vstack((self.Y, np.atleast_2d(y)))

        if hasattr(gp, 'update_cache'):
            gp.update_cache()
        self.update_feed_dict()


class ScipyDelaunay(spatial.Delaunay):
    """
    A dummy triangulation on a regular grid, very inefficient.

    Warning: The internal indexing is different from the one used in our
    implementation!

    Parameters
    ----------
    limits: array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    """

    def __init__(self, limits, num_points):
        self.numpoints = num_points
        self.limits = np.asarray(limits, dtype=config.np_dtype)
        params = [np.linspace(limit[0], limit[1], n) for limit, n in
                  zip(limits, num_points)]
        output = np.meshgrid(*params)
        points = np.array([par.ravel() for par in output]).T
        super(ScipyDelaunay, self).__init__(points)


class DimensionError(Exception):
    pass


class GridWorld(object):
    """Base class for function approximators on a regular grid.

    Parameters
    ----------
    limits: 2d array-like
        A list of limits. For example, [(x_min, x_max), (y_min, y_max)]
    num_points: 1d array-like
        The number of points with which to grid each dimension.

    """

    def __init__(self, limits, num_points):
        """Initialization, see `GridWorld`."""
        super(GridWorld, self).__init__()

        self.limits = np.atleast_2d(limits).astype(config.np_dtype)
        num_points = np.broadcast_to(num_points, len(self.limits))
        self.num_points = num_points.astype(np.int, copy=False)

        if np.any(self.num_points < 2):
            raise DimensionError('There must be at least 2 points in each '
                                 'dimension.')

        # Compute offset and unit hyperrectangle
        self.offset = self.limits[:, 0]
        self.unit_maxes = ((self.limits[:, 1] - self.offset)
                           / (self.num_points - 1)).astype(config.np_dtype)
        self.offset_limits = np.stack((np.zeros_like(self.limits[:, 0]),
                                       self.limits[:, 1] - self.offset),
                                      axis=1)

        # Statistics about the grid
        self.discrete_points = [np.linspace(low, up, n, dtype=config.np_dtype)
                                for (low, up), n in zip(self.limits,
                                                        self.num_points)]

        self.nrectangles = np.prod(self.num_points - 1)
        self.nindex = np.prod(self.num_points)

        self.ndim = len(self.limits)
        self._all_points = None

    @property
    def all_points(self):
        """Return all the discrete points of the discretization.

        Returns
        -------
        points : ndarray
            An array with all the discrete points with size
            (self.nindex, self.ndim).

        """
        if self._all_points is None:
            mesh = np.meshgrid(*self.discrete_points, indexing='ij')
            points = np.column_stack(col.ravel() for col in mesh)
            self._all_points = points.astype(config.np_dtype)

        return self._all_points

    def __len__(self):
        """Return the number of points in the discretization."""
        return self.nindex

    def sample_continuous(self, num_samples):
        """Sample uniformly at random from the continuous domain.

        Parameters
        ----------
        num_samples : int

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        limits = self.limits
        rand = np.random.uniform(0, 1, size=(num_samples, self.ndim))
        return rand * np.diff(limits, axis=1).T + self.offset

    def sample_discrete(self, num_samples, replace=False):
        """Sample uniformly at random from the discrete domain.

        Parameters
        ----------
        num_samples : int
        replace : bool, optional
            Whether to sample with replacement.

        Returns
        -------
        points : ndarray
            Random points on the continuous rectangle.

        """
        idx = np.random.choice(self.nindex, size=num_samples, replace=replace)
        return self.index_to_state(idx)

    def _check_dimensions(self, states):
        """Raise an error if the states have the wrong dimension.

        Parameters
        ----------
        states : ndarray

        """
        if not states.shape[1] == self.ndim:
            raise DimensionError('the input argument has the wrong '
                                 'dimensions.')

    def _center_states(self, states, clip=True):
        """Center the states to the interval [0, x].

        Parameters
        ----------
        states : np.array
        clip : bool, optinal
            If False the data is not clipped to lie within the limits.

        Returns
        -------
        offset_states : ndarray

        """
        states = np.atleast_2d(states).astype(config.np_dtype)
        states = states - self.offset[None, :]
        if clip:
            np.clip(states,
                    self.offset_limits[:, 0] + 2 * _EPS,
                    self.offset_limits[:, 1] - 2 * _EPS,
                    out=states)
        return states

    def index_to_state(self, indices):
        """Convert indices to physical states.

        Parameters
        ----------
        indices : ndarray (int)
            The indices of points on the discretization.

        Returns
        -------
        states : ndarray
            The states with physical units that correspond to the indices.

        """
        indices = np.atleast_1d(indices)
        ijk_index = np.vstack(np.unravel_index(indices, self.num_points)).T
        ijk_index = ijk_index.astype(config.np_dtype)
        return ijk_index * self.unit_maxes + self.offset

    def state_to_index(self, states):
        """Convert physical states to indices.

        Parameters
        ----------
        states: ndarray
            Physical states on the discretization.

        Returns
        -------
        indices: ndarray (int)
            The indices that correspond to the physical states.

        """
        states = np.atleast_2d(states)
        self._check_dimensions(states)
        states = np.clip(states, self.limits[:, 0], self.limits[:, 1])
        states = (states - self.offset) * (1. / self.unit_maxes)
        ijk_index = np.rint(states).astype(np.int32)
        return np.ravel_multi_index(ijk_index.T, self.num_points)

    def state_to_rectangle(self, states):
        """Convert physical states to its closest rectangle index.

        Parameters
        ----------
        states : ndarray
            Physical states on the discretization.

        Returns
        -------
        rectangles : ndarray (int)
            The indices that correspond to rectangles of the physical states.

        """
        ind = []
        for i, (discrete, num_points) in enumerate(zip(self.discrete_points,
                                                       self.num_points)):
            idx = np.digitize(states[:, i], discrete)
            idx -= 1
            np.clip(idx, 0, num_points - 2, out=idx)

            ind.append(idx)
        return np.ravel_multi_index(ind, self.num_points - 1)

    def rectangle_to_state(self, rectangles):
        """
        Convert rectangle indices to the states of the bottem-left corners.

        Parameters
        ----------
        rectangles : ndarray (int)
            The indices of the rectangles

        Returns
        -------
        states : ndarray
            The states that correspond to the bottom-left corners of the
            corresponding rectangles.

        """
        rectangles = np.atleast_1d(rectangles)
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        ijk_index = ijk_index.astype(config.np_dtype)
        return (ijk_index.T * self.unit_maxes) + self.offset

    def rectangle_corner_index(self, rectangles):
        """Return the index of the bottom-left corner of the rectangle.

        Parameters
        ----------
        rectangles: ndarray
            The indices of the rectangles.

        Returns
        -------
        corners : ndarray (int)
            The indices of the bottom-left corners of the rectangles.

        """
        ijk_index = np.vstack(np.unravel_index(rectangles,
                                               self.num_points - 1))
        return np.ravel_multi_index(np.atleast_2d(ijk_index),
                                    self.num_points)


class PiecewiseConstant(DeterministicFunction):
    """A piecewise constant function approximator.

    Parameters
    ----------
    discretization : instance of discretization
        For example, an instance of `GridWorld`.
    vertex_values: arraylike, optional
        A 2D array with the values at the vertices of the grid on each row.

    """

    def __init__(self, discretization, vertex_values=None):
        """Initialization, see `PiecewiseConstant`."""
        super(PiecewiseConstant, self).__init__()

        self.discretization = discretization
        self._parameters = None
        self.parameters = vertex_values

        self.input_dim = discretization.ndim

    @property
    def output_dim(self):
        """Return the output dimensions of the function."""
        if self.parameters is not None:
            return self.parameters.shape[1]

    @property
    def parameters(self):
        """Return the vertex values."""
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        """Set the vertex values."""
        if values is None:
            self._parameters = values
        else:
            self._parameters = np.asarray(values).reshape(self.nindex, -1)

    @property
    def limits(self):
        """Return the discretization limits."""
        return self.discretization.limits

    @property
    def nindex(self):
        """Return the number of discretization indices."""
        return self.discretization.nindex

    def build_evaluation(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : ndarray
            The function values at the points.

        """
        nodes = self.discretization.state_to_index(points)
        return self.parameters[nodes]

    def parameter_derivative(self, points):
        """
        Obtain function values at points from triangulation.

        This function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : ndarray
            A 2d array where each row represents one point.

        Returns
        -------
        values
            A sparse matrix B so that evaluate(points) = B.dot(parameters).

        """
        npoints = len(points)
        weights = np.ones(npoints, dtype=np.int)
        rows = np.arange(npoints)
        cols = self.discretization.state_to_index(points)
        return sparse.coo_matrix((weights, (rows, cols)),
                                 shape=(npoints, self.nindex))

    def gradient(self, points):
        """Return the gradient.

        The gradient is always zero for piecewise constant functions!

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        gradient : ndarray
            The function gradient at the points.

        """
        return np.broadcast_to(0, (len(points), self.input_dim))


class _Delaunay1D(object):
    """A simple class that behaves like scipy.Delaunay for 1D inputs.

    Parameters
    ----------
    points : ndarray
        Coordinates of points to triangulate, shape (2, 1).

    """

    def __init__(self, points):
        """Initialization, see `_Delaunay1D`."""
        if points.shape[1] > 1:
            raise AttributeError('This only works for 1D inputs.')
        if points.shape[0] > 2:
            raise AttributeError('This only works for two points')

        self.points = points
        self.nsimplex = len(points) - 1

        self._min = np.min(points)
        self._max = np.max(points)

        self.simplices = np.array([[0, 1]])

    def find_simplex(self, points):
        """Find the simplices containing the given points.

        Parameters
        ----------
        points : ndarray
            2D array of coordinates of points for which to find simplices.

        Returns
        -------
        indices : ndarray
            Indices of simplices containing each point. Points outside the
            triangulation get the value -1.

        """
        points = points.squeeze()
        out_of_bounds = points > self._max
        out_of_bounds |= points < self._min
        return np.where(out_of_bounds, -1, 0)


class _Triangulation(DeterministicFunction):
    """
    Efficient Delaunay triangulation on regular grids.

    This class is a wrapper around scipy.spatial.Delaunay for regular grids. It
    splits the space into regular hyperrectangles and then computes a Delaunay
    triangulation for only one of them. This single triangulation is then
    generalized to other hyperrectangles, without ever maintaining the full
    triangulation for all individual hyperrectangles.

    Parameters
    ----------
    discretization : instance of discretization
        For example, an instance of `GridWorld`.
    vertex_values: arraylike, optional
        A 2D array with the values at the vertices of the grid on each row.
    project: bool, optional
        Whether to project points onto the limits.

    """

    def __init__(self, discretization, vertex_values=None, project=False):
        """Initialization."""
        super(_Triangulation, self).__init__()

        self.discretization = discretization
        self.input_dim = discretization.ndim

        self._parameters = None
        self.parameters = vertex_values

        disc = self.discretization

        # Get triangulation
        if len(disc.limits) == 1:
            corners = np.array([[0], disc.unit_maxes])
            self.triangulation = _Delaunay1D(corners)
        else:
            product = cartesian(*np.diag(disc.unit_maxes))
            hyperrectangle_corners = np.array(list(product),
                                              dtype=config.np_dtype)
            self.triangulation = spatial.Delaunay(hyperrectangle_corners)
        self.unit_simplices = self._triangulation_simplex_indices()

        # Some statistics about the triangulation
        self.nsimplex = self.triangulation.nsimplex * disc.nrectangles

        # Parameters for the hyperplanes of the triangulation
        self.hyperplanes = None
        self._update_hyperplanes()

        self.project = project

    @property
    def output_dim(self):
        """Return the output dimensions of the function."""
        if self.parameters is not None:
            return self.parameters.shape[1]

    @property
    def parameters(self):
        """Return the vertex values."""
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        """Set the vertex values."""
        if values is None:
            self._parameters = values
        else:
            values = np.asarray(values).reshape(self.nindex, -1)
            self._parameters = values

    @property
    def limits(self):
        """Return the discretization limits."""
        return self.discretization.limits

    @property
    def nindex(self):
        """Return the number of discretization indices."""
        return self.discretization.nindex

    def _triangulation_simplex_indices(self):
        """Return the simplex indices in our coordinates.

        Returns
        -------
        simplices: ndarray (int)
            The simplices array in our extended coordinate system.

        Notes
        -----
        This is only used once in the initialization.

        """
        disc = self.discretization
        simplices = self.triangulation.simplices
        new_simplices = np.empty_like(simplices)

        # Convert the points to out indices
        index_mapping = disc.state_to_index(self.triangulation.points +
                                            disc.offset)

        # Replace each index with out new_index in index_mapping
        for i, new_index in enumerate(index_mapping):
            new_simplices[simplices == i] = new_index
        return new_simplices

    def _update_hyperplanes(self):
        """Compute the simplex hyperplane parameters on the triangulation."""
        self.hyperplanes = np.empty((self.triangulation.nsimplex,
                                     self.input_dim, self.input_dim),
                                    dtype=config.np_dtype)

        # Use that the bottom-left rectangle has the index zero, so that the
        # index numbers of scipy correspond to ours.
        for i, simplex in enumerate(self.unit_simplices):
            simplex_points = self.discretization.index_to_state(simplex)
            self.hyperplanes[i] = np.linalg.inv(simplex_points[1:] -
                                                simplex_points[:1])

    def find_simplex(self, points):
        """Find the simplices corresponding to points.

        Parameters
        ----------
        points : 2darray

        Returns
        -------
        simplices : np.array (int)
            The indices of the simplices

        """
        disc = self.discretization
        rectangles = disc.state_to_rectangle(points)

        # Convert to unit coordinates
        points = disc._center_states(points, clip=True)

        # Convert to basic hyperrectangle coordinates and find simplex
        unit_coordinates = points % disc.unit_maxes
        simplex_ids = self.triangulation.find_simplex(unit_coordinates)
        simplex_ids = np.atleast_1d(simplex_ids)

        # Adjust for the hyperrectangle index
        simplex_ids += rectangles * self.triangulation.nsimplex

        return simplex_ids

    def simplices(self, indices):
        """Return the simplices corresponding to the simplex index.

        Parameters
        ----------
        indices : ndarray
            The indices of the simpleces

        Returns
        -------
        simplices : ndarray
            Each row consists of the indices of the simplex corners.

        """
        # Get the indices inside the unit rectangle
        unit_indices = np.remainder(indices, self.triangulation.nsimplex)
        simplices = self.unit_simplices[unit_indices].copy()

        # Shift indices to corresponding rectangle
        rectangles = np.floor_divide(indices, self.triangulation.nsimplex)
        corner_index = self.discretization.rectangle_corner_index(rectangles)

        if simplices.ndim > 1:
            corner_index = corner_index[:, None]

        simplices += corner_index
        return simplices

    def _get_weights(self, points):
        """Return the linear weights associated with points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point

        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indices of the simplices associated with each points

        """
        disc = self.discretization
        simplex_ids = self.find_simplex(points)

        simplices = self.simplices(simplex_ids)
        origins = disc.index_to_state(simplices[:, 0])

        # Get hyperplane equations
        simplex_ids %= self.triangulation.nsimplex
        hyperplanes = self.hyperplanes[simplex_ids]

        # Some numbers for convenience
        nsimp = self.input_dim + 1
        npoints = len(points)

        if self.project:
            points = np.clip(points, disc.limits[:, 0], disc.limits[:, 1])

        weights = np.empty((npoints, nsimp), dtype=config.np_dtype)

        # Pre-multiply each hyperplane by (point - origin)
        offset = points - origins
        np.sum(offset[:, :, None] * hyperplanes, axis=1, out=weights[:, 1:])

        # The weights have to add up to one
        weights[:, 0] = 1 - np.sum(weights[:, 1:], axis=1)

        return weights, simplices

    def build_evaluation(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : ndarray
            The function values at the points.

        """
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights(points)

        # Return function values
        parameter_vector = self.parameters[simplices]

        # Broadcast the weights along output dimensions
        return np.sum(weights[:, :, None] * parameter_vector, axis=1)

    def parameter_derivative(self, points):
        """
        Obtain function values at points from triangulation.

        This function returns a sparse matrix that, when multiplied
        with the vector with all the function values on the vertices,
        returns the function values at points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point.

        Returns
        -------
        values
            A sparse matrix B so that evaluate(points) = B.dot(parameters).

        """
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights(points)
        # Construct sparse matrix for optimization

        nsimp = self.input_dim + 1
        npoints = len(simplices)
        # Indices of constraints (nsimp points per simplex, so we have nsimp
        # values in each row; one for each simplex)
        rows = np.repeat(np.arange(len(points)), nsimp)
        cols = simplices.ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(npoints, self.discretization.nindex))

    def _get_weights_gradient(self, points=None, indices=None):
        """Return the linear gradient weights associated with points.

        Parameters
        ----------
        points : ndarray
            Each row represents one point.
        indices : ndarray
            Each row represents one index. Ignored if points

        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        simplices : ndarray
            The indices of the simplices associated with each points

        """
        if points is None:
            simplex_ids = np.atleast_1d(indices)
        elif indices is None:
            simplex_ids = self.find_simplex(points)
        else:
            raise TypeError('Need to provide at least one input argument.')
        simplices = self.simplices(simplex_ids)

        # Get hyperplane equations
        simplex_ids %= self.triangulation.nsimplex

        # Some numbers for convenience
        nsimp = self.input_dim + 1
        npoints = len(simplex_ids)

        # weights
        weights = np.empty((npoints, self.input_dim, nsimp),
                           dtype=config.np_dtype)

        weights[:, :, 1:] = self.hyperplanes[simplex_ids]
        weights[:, :, 0] = -np.sum(weights[:, :, 1:], axis=2)
        return weights, simplices

    def gradient(self, points):
        """Return the gradient.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        gradient : ndarray
            The function gradient at the points. A 3D array with the gradient
            at the i-th data points for the j-th output with regard to the k-th
            dimension stored at (i, j, k). The j-th dimension is squeezed out
            for 1D functions.

        """
        points = np.atleast_2d(points)
        weights, simplices = self._get_weights_gradient(points)
        # Return function values if desired
        res = np.einsum('ijk,ikl->ilj', weights, self.parameters[simplices, :])
        if res.shape[1] == 1:
            res = res.squeeze(axis=1)
        return res

    def gradient_parameter_derivative(self, points=None, indices=None):
        """
        Return the gradients at the respective points.

        This function returns a sparse matrix that, when multiplied
        with the vector of all the function values on the vertices,
        returns the gradients. Note that after the product you have to call
        ```np.reshape(grad, (ndim, -1))``` in order to obtain a proper
        gradient matrix.

        Parameters
        ----------
        points : ndarray
            Each row contains one state at which to evaluate the gradient.
        indices : ndarray
            The simplex indices. Ignored if points are provided.

        Returns
        -------
        gradient : scipy.sparse.coo_matrix
            A sparse matrix so that
            `grad(points) = B.dot(vertex_val).reshape(ndim, -1)` corresponds
            to the true gradients

        """
        weights, simplices = self._get_weights_gradient(points=points,
                                                        indices=indices)

        # Some numbers for convenience
        nsimp = self.input_dim + 1
        npoints = len(simplices)

        # Construct sparse matrix for optimization

        # Indices of constraints (ndim gradients for each point, which each
        # depend on the nsimp vertices of the simplex.
        rows = np.repeat(np.arange(npoints * self.input_dim), nsimp)
        cols = np.tile(simplices, (1, self.input_dim)).ravel()

        return sparse.coo_matrix((weights.ravel(), (rows, cols)),
                                 shape=(self.input_dim * npoints,
                                        self.discretization.nindex))


class Triangulation(DeterministicFunction):
    """Efficient Delaunay triangulation on regular grid.

    This is a tensorflow wrapper around a numpy implementation.

    This class is a wrapper around scipy.spatial.Delaunay for regular grids. It
    splits the space into regular hyperrectangles and then computes a Delaunay
    triangulation for only one of them. This single triangulation is then
    generalized to other hyperrectangles, without ever maintaining the full
    triangulation for all individual hyperrectangles.

    Parameters
    ----------
    discretization : instance of discretization
        For example, an instance of `GridWorld`.
    vertex_values : arraylike, optional
        A 2D array with the values at the vertices of the grid on each row.
        Is converted into a tensorflow variable.
    project : bool, optional
        Whether to project points onto the limits.
    name : string
        The tensorflow scope for all methods.

    """

    def __init__(self, discretization, vertex_values, project=False,
                 name='triangulation'):
        """Initialization."""
        super(Triangulation, self).__init__(name=name)

        with tf.variable_scope(self.scope_name):
            self.tri = _Triangulation(discretization,
                                      project=project)

            # Make sure the variable has the correct size
            if not isinstance(vertex_values, tf.Variable):
                self.tri.parameters = vertex_values
                vertex_values = self.tri.parameters.astype(config.np_dtype)
            vertex_values = tf.Variable(vertex_values,
                                        name='vertex_values')

            # Initialize parameters
            sess = tf.get_default_session()
            if sess is not None:
                init = tf.variables_initializer(self.parameters)
                tf.get_default_session().run(init)

            self.input_dim = self.tri.input_dim
            self.output_dim = self.tri.output_dim

    @property
    def project(self):
        """Getter for the project parameter."""
        return self.tri.project

    @project.setter
    def project(self, value):
        """Setter for the project parameter."""
        self.tri.project = value

    @property
    def discretization(self):
        """Getter for the discretization."""
        return self.tri.discretization

    @property
    def nindex(self):
        """Return the number of parameters."""
        return self.tri.nindex

    @make_tf_fun([config.dtype, config.dtype, tf.int64], stateful=False)
    def _get_hyperplanes(self, points):
        """Return the linear weights associated with points.

        Parameters
        ----------
        points : 2d array
            Each row represents one point

        Returns
        -------
        weights : ndarray
            An array that contains the linear weights for each point.
        hyperplanes : ndarray
            The corresponding hyperplane objects.
        simplices : ndarray
            The indices of the simplices associated with each points

        """
        simplex_ids = self.tri.find_simplex(points)

        simplices = self.tri.simplices(simplex_ids).astype(np.int64)
        origins = self.tri.discretization.index_to_state(simplices[:, 0])

        # Get hyperplane equations
        simplex_ids %= self.tri.triangulation.nsimplex
        hyperplanes = self.tri.hyperplanes[simplex_ids]

        # Pre-multiply each hyperplane by (point - origin)
        return origins, hyperplanes, simplices

    def build_evaluation(self, points):
        """Evaluate using tensorflow."""
        # Get the appropriate hyperplane
        origins, hyperplanes, simplices = self._get_hyperplanes(points)

        # Project points onto the grid of triangles.
        if self.project:
            clip_min = self.tri.limits[:, 0]
            clip_max = self.tri.limits[:, 1]

            # Broadcasting in tf.clip_by_value not available in
            # TensorFlow >= 1.6.0
            points = tf.minimum(tf.maximum(points, clip_min), clip_max)

        # Compute weights (barycentric coordinates)
        offset = points - origins
        w1 = tf.reduce_sum(offset[:, :, None] * hyperplanes, axis=1)
        w0 = 1 - tf.reduce_sum(w1, axis=1, keepdims=True)
        weights = tf.concat((w0, w1), axis=1)

        # Collect the value on the vertices
        parameter_vector = tf.gather(self.parameters[0],
                                     indices=simplices,
                                     validate_indices=False)

        # Compute the values
        return tf.reduce_sum(weights[:, :, None] * parameter_vector, axis=1)

    @make_tf_fun([config.dtype], stateful=False)
    def _get_gradients(self, points, parameters):
        self.tri.parameters = parameters
        return self.tri.gradient(points)

    @use_parent_scope
    @with_scope('derivative')
    def gradient(self, points):
        """Compute derivatives using tensorflow."""
        return self._get_gradients(points, self.parameters[0])[0]


class QuadraticFunction(DeterministicFunction):
    """A quadratic function.

    values(x) = x.T P x

    Parameters
    ----------
    matrix : np.array
        2d cost matrix for lyapunov function.

    """

    def __init__(self, matrix, name='quadratic'):
        """Initialization, see `QuadraticLyapunovFunction`."""
        super(QuadraticFunction, self).__init__(name=name)

        self.matrix = np.atleast_2d(matrix).astype(config.np_dtype)
        self.ndim = self.matrix.shape[0]
        # with tf.variable_scope(self.scope_name):
        #     self.matrix = tf.Variable(self.matrix)

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Like evaluate, but returns a tensorflow tensor instead."""
        linear_form = tf.matmul(points, self.matrix)
        quadratic = linear_form * points
        return tf.reduce_sum(quadratic, axis=1, keepdims=True)

    def gradient(self, points):
        """Return the gradient of the function."""
        return tf.matmul(points, self.matrix + self.matrix.T)


class LinearSystem(DeterministicFunction):
    """A linear system.

    y = A_1 * x + A_2 * x_2 ...

    Parameters
    ----------
    *matrices : list
        Can specify an arbitrary amount of matrices for the linear system. Each
        is multiplied by the corresponding state that is passed to evaluate.

    """

    def __init__(self, matrices, name='linear_system'):
        """Initialize."""
        super(LinearSystem, self).__init__(name=name)
        fun = lambda x: np.atleast_2d(x).astype(config.np_dtype)
        self.matrix = np.hstack(map(fun, matrices))

        self.output_dim, self.input_dim = self.matrix.shape

    @concatenate_inputs(start=1)
    def build_evaluation(self, points):
        """Return the function values.

        Parameters
        ----------
        points : ndarray
            The points at which to evaluate the function. One row for each
            data points.

        Returns
        -------
        values : tf.Tensor
            A 2D array with the function values at the points.

        """
        return tf.matmul(points, self.matrix.T, transpose_b=False)


@with_scope('sample_gp_function')
def sample_gp_function(discretization, gpfun, number=1, return_function=True):
    """
    Sample a function from a gp with corresponding kernel within its bounds.

    Parameters
    ----------
    discretization : ndarray
        The discretization on which to draw a sample from the GP. Can be
        obtained, for example, from GridWorld.all_points.
    gpfun : instance of safe_learning.GaussianProcess
        The GP from which to draw a sample.
    number : int
        The number of functions to sample.
    return_function : bool, optional
        Whether to return a function or the sampled data only.

    Returns
    -------
    function : list of functions or ndarray
        function(x, noise=True)
        A function that takes as inputs new locations x to be evaluated and
        returns the corresponding noisy function values as a tensor. If
        noise=False is set the true function values are returned (useful for
        plotting).

    """
    if isinstance(discretization, GridWorld):
        discretization = discretization.all_points

    gp = gpfun.gaussian_process

    with gp.tf_mode():
        mean, cov = gp.build_predict(discretization, full_cov=True)

    # Evaluate
    sess = tf.get_default_session()
    mean, cov = sess.run([mean, cov], feed_dict=gpfun.feed_dict)

    # Turn mean and covariance into 1D and 2D arrays
    mean = mean.squeeze(-1)
    cov = cov.squeeze(-1)

    # Make sure the covariance is positive definite
    cov += np.eye(len(cov)) * 1E-8

    # Draw a sample
    output = np.random.multivariate_normal(mean, cov, size=number)

    if not return_function:
        return output

    # cholesky
    cho_factor = linalg.cho_factor(cov, lower=True)

    @concatenate_inputs(start=1)
    def gp_sample(alpha, x, noise=True):
        with gp.tf_mode():
            k = gp.kern.K(x, discretization)
            y = gp.mean_function(x) + tf.matmul(k, alpha)
            if noise:
                y += (tf.sqrt(gp.likelihood.variance)
                      * tf.random_normal(tf.shape(y), dtype=tf.float64))
        return y

    # Now let's plug in the alpha to generate samples
    functions = []
    for i in range(number):
        alpha = linalg.cho_solve(cho_factor, output[[i], :].T)
        fun = partial(gp_sample, alpha)

        # Attach the feed_dict for ease of use
        fun.feed_dict = gpfun.feed_dict

        functions.append(fun)

    return functions


class NeuralNetwork(DeterministicFunction):
    """A simple neural network.

    The neural network also exposes its Lipschitz constant as
    `NeuralNetwork.lipschitz`.

    Parameters
    ----------
    layers : list
        A list of layer sizes, [l1, l2, l3, .., ln]. l1 corresponds to the
        input dimension of the neural network, while ln is the output
        dimension.
    nonlinearities : list
        A list of nonlinearities applied after each layer. Can be None if no
        nonlinearity should be applied.
    output_scale : float, optional
        A constant scaling factor applied to the neural network output.
    use_bias : bool, optional
        A boolean determining whether bias terms are included in each hidden
        layer; bias terms are never used in the output layer.
    name : string, optional

    """

    def __init__(self, layers, nonlinearities, output_scale=1., use_bias=True,
                 name='neural_network'):
        """Initialization, see `NeuralNetwork`."""
        super(NeuralNetwork, self).__init__(name=name)

        self.layers = layers
        self.nonlinearities = nonlinearities
        self.output_scale = output_scale
        self.use_bias = use_bias

        self.input_dim = layers[0]
        self.output_dim = layers[-1]

    def build_evaluation(self, points):
        """Build the evaluation graph."""
        net = points
        if isinstance(net, np.ndarray):
            net = tf.constant(net)

        initializer = tf.contrib.layers.xavier_initializer()

        for i, (layer, activation) in enumerate(zip(self.layers[:-1],
                                                    self.nonlinearities[:-1])):
            net = tf.layers.dense(net,
                                  units=layer,
                                  activation=activation,
                                  use_bias=self.use_bias,
                                  kernel_initializer=initializer,
                                  name='layer_{}'.format(i))

        # Output layer
        net = tf.layers.dense(net,
                              units=self.layers[-1],
                              activation=self.nonlinearities[-1],
                              use_bias=False,
                              kernel_initializer=initializer,
                              name='output')

        # Scale output range
        out = tf.multiply(net, self.output_scale, name='output_scale')
        return out

    def _parameter_iter(self):
        """Iterate over parameters in (W, b) tuples."""
        # By defining parameters as an iterable zip(parameters, parameters)
        # returns the next two elements as a group.
        parameters = iter(self.parameters)
        for W, b in zip(parameters, parameters):
            yield W, b

        # Yield parameters of the output layer
        yield self.parameters[-1], None

    @use_parent_scope
    @with_scope('lipschitz_constant')
    def lipschitz(self):
        """Return the Lipschitz constant as a Tensor.

        This assumes that only contractive nonlinearities are used! Examples
        are ReLUs and Sigmoids.

        Returns
        -------
        lipschitz : Tensor
            The Lipschitz constant of the neural network.

        """
        lipschitz = tf.constant(1, config.dtype)

        for W, b in self._parameter_iter():
            # lipschitz *= tf.reduce_max(tf.svd(W, compute_uv=False))
            lipschitz *= tf.reduce_max(self._svd(W))

        return lipschitz

    @staticmethod
    def _svd(A, name=None):
        """Tensorflow svd with gradient.

        Parameters
        ----------
        A : Tensor
            The matrix for which to compute singular values.
        name : string, optional

        Returns
        -------
        s : Tensor
            The singular values of A.

        """
        S0, U0, V0 = map(tf.stop_gradient,
                         tf.svd(A, full_matrices=True, name=name))
        # A = U * S * V.T
        # S = inv(U) * A * inv(V.T) = U.T * A * V  (orthogonal matrices)
        S = tf.matmul(U0, tf.matmul(A, V0),
                      transpose_a=True)
        return tf.matrix_diag_part(S)
