"""Unit tests for the functions file."""

from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal, assert_allclose
import pytest
import numpy as np
import tensorflow as tf

from safe_learning.functions import (_Triangulation, Triangulation,
                                     ScipyDelaunay, GridWorld,
                                     PiecewiseConstant, DeterministicFunction,
                                     UncertainFunction, QuadraticFunction,
                                     DimensionError, GPRCached,
                                     GaussianProcess, NeuralNetwork)
from safe_learning.utilities import concatenate_inputs

try:
    import gpflow
except ImportError:
    gpflow = None


class TestFunction(object):
    """Test the function class."""

    @pytest.fixture(scope='class')
    def testing_class(self):
        class A(DeterministicFunction):
            def __init__(self, value, name='a'):
                super(A, self).__init__()
                with tf.variable_scope(self.scope_name):
                    self.variable = tf.Variable(value)
                    sess = tf.get_default_session()
                    sess.run(tf.variables_initializer([self.variable]))

            def build_evaluation(self, point):
                return self.variable * point

        sess = tf.Session()
        return A, sess

    def test_class(self, testing_class):
        """Test that the class is working."""
        A, sess = testing_class
        with sess.as_default():
            a = A(2.)
            input = np.array(1.)

            output = a(input)
            assert_allclose(2. * input, output.eval())

            # Test double output
            output2 = a(input)
            assert_allclose(2. * input, output2.eval())

    def test_add(self, testing_class):
        """Test adding functions."""
        A, sess = testing_class
        with sess.as_default():
            a1 = A(3.)
            a2 = A(2.)

            a = a1 + a2

            input = np.array(1.)
            output = a(input)

            assert_allclose(5. * input, output.eval())

            assert a1.parameters[0] in a.parameters
            assert a2.parameters[0] in a.parameters

    def test_mult(self, testing_class):
        """Test multiplying functions."""
        A, sess = testing_class
        with sess.as_default():
            a1 = A(3.)
            a2 = A(2.)

            a = a1 * a2

            input = np.array(1.)
            output = a(input)

            assert_allclose(6. * input, output.eval())

            assert a1.parameters[0] in a.parameters
            assert a2.parameters[0] in a.parameters

            # Test multiplying with constant
            a = a1 * 2.
            output = a(input)
            assert_allclose(6. * input, output.eval())

    def test_neg(self, testing_class):
        """Test multiplying functions."""
        A, sess = testing_class
        with sess.as_default():
            a = A(3.)
            b = -a

            input = np.array(2.)
            output = b(input)

            assert_allclose(-3. * input, output.eval())

            assert a.parameters[0] is b.parameters[0]

    def test_copy(self, testing_class):
        """Test copying."""
        A, sess = testing_class
        with sess.as_default():
            a = A(2.)
            b = A(3.)
            b.copy_parameters(a)

            p1 = a.parameters[0]
            p2 = b.parameters[0]

            assert p1.eval() == p2.eval()
            assert p1 is not p2


class TestDeterministicFuction(object):
    """Test the base class."""

    def test_errors(self):
        """Check notImplemented error."""
        f = DeterministicFunction()
        pytest.raises(NotImplementedError, f.build_evaluation, None)


class TestUncertainFunction(object):
    """Test the base class."""

    def test_errors(self):
        """Check notImplemented error."""
        f = UncertainFunction()
        pytest.raises(NotImplementedError, f.build_evaluation, None)

    def test_mean_function(self):
        """Test the conversion to a deterministic function."""
        f = UncertainFunction()
        f.build_evaluation = lambda x: (1, 2)
        fd = f.to_mean_function()
        assert(fd(None) == 1)


@pytest.mark.skipif(gpflow is None, reason='gpflow module not installed')
class TestGPRCached(object):
    """Test the GPR_cached class."""

    @pytest.fixture(scope="class")
    def gps(self):
        """Create cached and uncached gpflow models and GPy model."""
        x = np.array([[1, 0], [0, 1]], dtype=float)
        y = np.array([[0], [1]], dtype=float)
        kernel = gpflow.kernels.RBF(2)
        gp = gpflow.gpr.GPR(x, y, kernel)
        gp_cached = GPRCached(x, y, kernel)
        return gp, gp_cached

    def test_adding_data(self, gps):
        """Test that adding data works."""
        test_points = np.array([[0.9, 0.1], [3., 2]])

        gp, gp_cached = gps
        gpfun = GaussianProcess(gp)
        gpfun_cached = GaussianProcess(gp_cached)

        x = np.array([[1.2, 2.3]])
        y = np.array([[2.4]])

        gpfun.add_data_point(x, y)
        m1, v1 = gpfun(test_points)

        gpfun_cached.add_data_point(x, y)
        m2, v2 = gpfun_cached(test_points)

        feed_dict = gpfun.feed_dict.copy()
        feed_dict.update(gpfun_cached.feed_dict)

        with tf.Session() as sess:
            m1, v1, m2, v2 = sess.run([m1, v1, m2, v2], feed_dict=feed_dict)

        assert_allclose(m1, m2)
        assert_allclose(v1, v2)

    def test_predict_f(self, gps):
        """Make sure predictions is same as in uncached case."""
        # Note that this messes things up terribly due to caching. So this
        # must be the last test that we run.
        gp, gp_cached = gps
        test_points = np.array([[0.9, 0.1], [3., 2]])
        a1, b1 = gp_cached.predict_f(test_points)
        a2, b2 = gp.predict_f(test_points)
        assert_allclose(a1, a2)
        assert_allclose(b1, b2)


@pytest.mark.skipIf(gpflow is None, 'gpflow module not installed')
class Testgpflow(object):
    """Test the GaussianProcess function class."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Create GP model with gpflow and GPy."""
        with tf.Session() as sess:
            x = np.array([[1, 0], [0, 1]], dtype=float)
            y = np.array([[0], [1]], dtype=float)
            kernel = gpflow.kernels.RBF(2)
            gp = gpflow.gpr.GPR(x, y, kernel)
            yield sess, gp

    def test_evaluation(self, setup):
        """Make sure evaluation works."""
        test_points = np.array([[0.9, 0.1], [3., 2]])
        beta = 3.0
        sess, gp = setup

        ufun = GaussianProcess(gp, beta=beta)

        # Evaluate GP
        mean_1, error_1 = ufun(test_points)
        mean_1, error_1 = sess.run([mean_1, error_1],
                                   feed_dict=ufun.feed_dict)

        # Test multiple inputs
        mean_2, error_2 = ufun(test_points[:, [0]],
                               test_points[:, [1]])
        mean_2, error_2 = sess.run([mean_2, error_2], feed_dict=ufun.feed_dict)

        assert_allclose(mean_1, mean_2)
        assert_allclose(error_1, error_2)

    def test_new_data(self, setup):
        """Test adding data points to the GP."""
        test_points = np.array([[0.9, 0.1], [3., 2]])
        sess, gp = setup

        ufun = GaussianProcess(gp)

        x = np.array([[1.2, 2.3]])
        y = np.array([[2.4]])

        ufun.add_data_point(x, y)

        assert_allclose(ufun.X, np.array([[1, 0],
                                          [0, 1],
                                          [1.2, 2.3]]))
        assert_allclose(ufun.Y, np.array([[0], [1], [2.4]]))

        # Check prediction is correct after adding data (cholesky update)
        a1, b1 = ufun(test_points)
        a1, b1 = sess.run([a1, b1], feed_dict=ufun.feed_dict)

        a1_true = np.array([[0.16371139], [0.22048311]])
        b1_true = np.array([[1.37678679], [1.98183191]])
        assert_allclose(a1, a1_true)
        assert_allclose(b1, b1_true)


class TestQuadraticFunction(object):
    """Test the quadratic function."""

    def test_evaluate(self):
        """Setup testing environment for quadratic."""
        points = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]], dtype=np.float)
        P = np.array([[1., 0.1],
                      [0.2, 2.]])
        quad = QuadraticFunction(P)
        true_fval = np.array([[0., 2., 1., 3.3]]).T

        with tf.Session():
            tf_res = quad(points)
            res = tf_res.eval()

        assert_allclose(true_fval, res)


def test_scipy_delaunay():
    """Test the fake replacement for Scipy."""
    limits = [[-1, 1], [-1, 2]]
    num_points = [2, 6]
    discretization = GridWorld(limits, num_points)
    sp_delaunay = ScipyDelaunay(limits, num_points)
    delaunay = _Triangulation(discretization)

    assert_equal(delaunay.nsimplex, sp_delaunay.nsimplex)
    assert_equal(delaunay.input_dim, sp_delaunay.ndim)
    sp_delaunay.find_simplex(np.array([[0, 0]]))


class TestGridworld(object):
    """Test the general GridWorld definitions."""

    def test_dimensions_error(self):
        """Test dimension errors."""
        limits = [[-1.1, 1.5], [2.2, 2.4]]
        num_points = [7, 8]
        grid = GridWorld(limits, num_points)

        pytest.raises(DimensionError, grid._check_dimensions,
                      np.array([[1, 2, 3]]))

        pytest.raises(DimensionError, grid._check_dimensions,
                      np.array([[1]]))

    def test_index_state_conversion(self):
        """Test all index conversions."""
        limits = [[-1.1, 1.5], [2.2, 2.4]]
        num_points = [7, 8]
        grid = GridWorld(limits, num_points)

        # Forward and backwards convert all indeces
        indeces = np.arange(grid.nindex)
        states = grid.index_to_state(indeces)
        indeces2 = grid.state_to_index(states)
        assert_equal(indeces, indeces2)

        # test 1D input
        grid.state_to_index([0, 2.3])
        grid.index_to_state(1)

        # Test rectangles
        rectangles = np.arange(grid.nrectangles)
        states = grid.rectangle_to_state(rectangles)
        rectangles2 = grid.state_to_rectangle(states + grid.unit_maxes / 2)
        assert_equal(rectangles, rectangles2)

        rectangle = grid.state_to_rectangle(100 * np.ones((1, 2)))
        assert_equal(rectangle, grid.nrectangles - 1)

        rectangle = grid.state_to_rectangle(-100 * np.ones((1, 2)))
        assert_equal(rectangle, 0)

        # Test rectangle corners
        corners = grid.rectangle_corner_index(rectangles)
        corner_states = grid.rectangle_to_state(rectangles)
        corners2 = grid.state_to_index(corner_states)
        assert_equal(corners, corners2)

        # Test point outside grid
        test_point = np.array([[-1.2, 2.]])
        index = grid.state_to_index(test_point)
        assert_equal(index, 0)

    def test_integer_numpoints(self):
        """Check integer numpoints argument."""
        grid = GridWorld([[1, 2], [3, 4]], 2)
        assert_equal(grid.num_points, np.array([2, 2]))

    def test_0d(self):
        """Check that initialization works for 1d-discretization."""
        grid = GridWorld([[0, 1]], 3)

        test = np.array([[0.1, 0.4, 0.9]]).T
        res = np.array([0, 1, 2])
        assert_allclose(grid.state_to_index(test), res)

        res = np.array([0, 0, 1])
        assert_allclose(grid.state_to_rectangle(test), res)
        assert_allclose(grid.rectangle_to_state(res), res[:, None] * 0.5)


class TestConcatenateDecorator(object):
    """Test the concatenate_input decorator."""

    @concatenate_inputs(start=1)
    def fun(self, x):
        """Test function."""
        return x

    def test_concatenate_numpy(self):
        """Test concatenation of inputs for numpy."""
        x = np.arange(4).reshape(2, 2)
        y = x + 4
        true_res = np.hstack((x, y))
        res = self.fun(x, y)
        assert_allclose(res, true_res)
        assert_allclose(self.fun(x), x)

    def test_concatenate_tensorflow(self):
        """Test concatenation of inputs for tensorflow."""
        x_data = np.arange(4).reshape(2, 2).astype(np.float32)
        true_res = np.hstack((x_data, x_data + 4))
        x = tf.placeholder(dtype=tf.float32, shape=[2, 2])
        y = x + 4

        fun_x = self.fun(x)
        fun_xy = self.fun(x, y)

        assert isinstance(fun_x, tf.Tensor)
        assert isinstance(fun_xy, tf.Tensor)

        with tf.Session() as sess:
            res_x, res_both = sess.run([fun_x, fun_xy],
                                       {x: x_data})

        assert_allclose(res_both, true_res)
        assert_allclose(res_x, x_data)


class TestPiecewiseConstant(object):
    """Test a piecewise constant function."""

    def test_init(self):
        """Test initialisation."""
        limits = [[-1, 1], [-1, 1]]
        npoints = 4
        discretization = GridWorld(limits, npoints)
        pwc = PiecewiseConstant(discretization, np.arange(16))
        assert_allclose(pwc.parameters, np.arange(16)[:, None])

    def test_evaluation(self):
        """Evaluation tests for piecewise constant function."""
        limits = [[-1, 1], [-1, 1]]
        npoints = 3
        discretization = GridWorld(limits, npoints)
        pwc = PiecewiseConstant(discretization)

        vertex_points = pwc.discretization.index_to_state(
            np.arange(pwc.nindex))
        vertex_values = np.sum(vertex_points, axis=1, keepdims=True)
        pwc.parameters = vertex_values

        test = pwc(vertex_points)
        assert_allclose(test, vertex_values)

        outside_point = np.array([[-1.5, -1.5]])
        test1 = pwc(outside_point)
        assert_allclose(test1, np.array([[-2]]))

        # Test constraint evaluation
        test2 = pwc.parameter_derivative(vertex_points)
        test2 = test2.toarray().dot(vertex_values)
        assert_allclose(test2, vertex_values)

    def test_gradient(self):
        """Test the gradient."""
        limits = [[-1, 1], [-1, 1]]
        npoints = 3
        discretization = GridWorld(limits, npoints)
        pwc = PiecewiseConstant(discretization)
        test_points = pwc.discretization.index_to_state(np.arange(pwc.nindex))
        gradient = pwc.gradient(test_points)
        assert_allclose(gradient, 0)


class TestTriangulationNumpy(object):
    """Test the generalized Delaunay triangulation in numpy."""

    def test_find_simplex(self):
        """Test the simplices on the grid."""
        limits = [[-1, 1], [-1, 2]]
        num_points = [3, 7]
        discretization = GridWorld(limits, num_points)
        delaunay = _Triangulation(discretization)

        # Test the basic properties
        assert_equal(delaunay.discretization.nrectangles, 2 * 6)
        assert_equal(delaunay.input_dim, 2)
        assert_equal(delaunay.nsimplex, 2 * 2 * 6)
        assert_equal(delaunay.discretization.offset, np.array([-1, -1]))
        assert_equal(delaunay.discretization.unit_maxes,
                     np.array([2, 3]) / (np.array(num_points) - 1))

        # test the simplex indices
        lower = delaunay.triangulation.find_simplex(np.array([0, 0])).squeeze()
        upper = 1 - lower

        test_points = np.array([[0, 0],
                                [0.9, 0.45],
                                [1.1, 0],
                                [1.9, 2.9]])

        test_points += np.array(limits)[:, 0]

        true_result = np.array([lower, upper, 6 * 2 + lower, 11 * 2 + upper])
        result = delaunay.find_simplex(test_points)

        assert_allclose(result, true_result)

        # Test the ability to find simplices
        simplices = delaunay.simplices(result)
        true_simplices = np.array([[0, 1, 7],
                                   [1, 7, 8],
                                   [7, 8, 14],
                                   [13, 19, 20]])
        assert_equal(np.sort(simplices, axis=1), true_simplices)

        # Test point ouside domain (should map to bottom left and top right)
        assert_equal(lower, delaunay.find_simplex(np.array([[-100., -100.]])))
        assert_equal(delaunay.nsimplex - 1 - lower,
                     delaunay.find_simplex(np.array([[100., 100.]])))

    def test_values(self):
        """Test the evaluation function."""
        eps = 1e-10

        discretization = GridWorld([[0, 1], [0, 1]], [2, 2])
        delaunay = _Triangulation(discretization)

        test_points = np.array([[0, 0],
                                [1 - eps, 0],
                                [0, 1 - eps],
                                [0.5 - eps, 0.5 - eps],
                                [0, 0.5],
                                [0.5, 0]])
        nodes = delaunay.discretization.state_to_index(np.array([[0, 0],
                                                       [1, 0],
                                                       [0, 1]]))

        H = delaunay.parameter_derivative(test_points).toarray()

        true_H = np.zeros((len(test_points), delaunay.nindex),
                          dtype=np.float)
        true_H[0, nodes[0]] = 1
        true_H[1, nodes[1]] = 1
        true_H[2, nodes[2]] = 1
        true_H[3, nodes[[1, 2]]] = 0.5
        true_H[4, nodes[[0, 2]]] = 0.5
        true_H[5, nodes[[0, 1]]] = 0.5

        assert_allclose(H, true_H, atol=1e-7)

        # Test value property
        values = np.random.rand(delaunay.nindex)
        delaunay.parameters = values
        v1 = H.dot(values)[:, None]
        v2 = delaunay(test_points)
        assert_allclose(v1, v2)

        # Test the projections
        test_point = np.array([[-0.5, -0.5]])
        delaunay.parameters = np.array([0, 1, 1, 1])
        unprojected = delaunay(test_point)
        delaunay.project = True
        projected = delaunay(test_point)

        assert_allclose(projected, np.array([[0]]))
        assert_allclose(unprojected, np.array([[-1]]))

    def test_multiple_dimensions(self):
        """Test delaunay in three dimensions."""
        limits = [[0, 1]] * 3
        discretization = GridWorld(limits, [2] * 3)
        delaunay = _Triangulation(discretization)
        assert_equal(delaunay.input_dim, 3)
        assert_equal(delaunay.discretization.nrectangles, 1)
        assert_equal(delaunay.nsimplex, np.math.factorial(3))

        corner_points = np.array([[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1],
                                  [0, 1, 1],
                                  [1, 1, 0],
                                  [1, 0, 1],
                                  [1, 1, 1]], dtype=np.float)

        values = np.sum(delaunay.discretization.index_to_state(np.arange(8)),
                        axis=1) / 3

        test_points = np.vstack((corner_points,
                                 np.array([[0, 0, 0.5],
                                           [0.5, 0, 0],
                                           [0, 0.5, 0],
                                           [0.5, 0.5, 0.5]])))
        corner_values = np.sum(corner_points, axis=1) / 3
        true_values = np.hstack((corner_values,
                                 np.array([1 / 6, 1 / 6, 1 / 6, 1 / 2])))

        delaunay.parameters = values
        result = delaunay(test_points)
        assert_allclose(result, true_values[:, None], atol=1e-5)

    def test_gradient(self):
        """Test the gradient_at function."""
        discretization = GridWorld([[0, 1], [0, 1]], [2, 2])
        delaunay = _Triangulation(discretization)

        points = np.array([[0, 0],
                           [1, 0],
                           [0, 1],
                           [1, 1]], dtype=np.int)
        nodes = delaunay.discretization.state_to_index(points)

        # Simplex with node values:
        # 3 - 1
        # | \ |
        # 1 - 2
        # --> x

        values = np.zeros(delaunay.nindex)
        values[nodes] = [1, 2, 3, 1]

        test_points = np.array([[0.01, 0.01],
                                [0.99, 0.99]])

        true_grad = np.array([[1, 2], [-2, -1]])

        # Construct true H (gradient as function of values)
        true_H = np.zeros((2 * delaunay.input_dim, delaunay.nindex))

        true_H[0, nodes[[0, 1]]] = [-1, 1]
        true_H[1, nodes[[0, 2]]] = [-1, 1]
        true_H[2, nodes[[2, 3]]] = [-1, 1]
        true_H[3, nodes[[1, 3]]] = [-1, 1]

        # Evaluate gradient with and without values
        H = delaunay.gradient_parameter_derivative(test_points).toarray()
        delaunay.parameters = values
        grad = delaunay.gradient(test_points)

        # Compare
        assert_allclose(grad, true_grad)
        assert_allclose(H, true_H)
        assert_allclose(true_grad,
                        H.dot(values).reshape(-1, delaunay.input_dim))

    def test_1d(self):
        """Test the triangulation for 1D inputs."""
        discretization = GridWorld([[0, 1]], 3)
        delaunay = _Triangulation(discretization, vertex_values=[0, 0.5, 0])
        vertex_values = delaunay.parameters

        test_points = np.array([[0, 0.2, 0.5, 0.6, 0.9, 1.]]).T
        test_point = test_points[[0], :]

        simplices = delaunay.find_simplex(test_points)
        true_simplices = np.array([0, 0, 1, 1, 1, 1])
        assert_allclose(simplices, true_simplices)
        assert_allclose(delaunay.find_simplex(test_point),
                        true_simplices[[0]])

        values = delaunay(test_points)
        true_values = np.array([0, 0.2, 0.5, 0.4, 0.1, 0])[:, None]
        assert_allclose(values, true_values)

        value_constraint = delaunay.parameter_derivative(test_points)
        values = value_constraint.toarray().dot(vertex_values)
        assert_allclose(values, true_values)

        gradient = delaunay.gradient(test_points)
        true_gradient = np.array([1, 1, -1, -1, -1, -1])[:, None]
        assert_allclose(gradient, true_gradient)

        gradient_deriv = delaunay.gradient_parameter_derivative(test_points)
        gradient = gradient_deriv.toarray().dot(vertex_values)
        assert_allclose(gradient.reshape(-1, 1), true_gradient)


class TestTriangulation(object):
    """Test the tensorflow wrapper around the numpy triangulation."""

    @pytest.fixture(scope="class")
    def setup(self):
        """Create testing environment."""
        with tf.Session(graph=tf.Graph()) as sess:
            npoints = 3

            discretization = GridWorld([[0, 1], [0, 1]], npoints)
            parameters = np.sum(discretization.all_points ** 2,
                                axis=1, keepdims=True)
            trinp = _Triangulation(discretization, vertex_values=parameters)

            tri = Triangulation(discretization, vertex_values=parameters)

            test_points = np.array([[-10, -10],
                                    [0.2, 0.7],
                                    [0, 0],
                                    [0, 1],
                                    [1, 1],
                                    [-0.2, 0.5],
                                    [0.43, 0.21]])

            sess.run(tf.global_variables_initializer())
            yield sess, tri, trinp, test_points

    def test_evaluate(self, setup):
        """Test the evaluations."""
        sess, tri, trinp, test_points = setup
        # with tf.Session() as sess:
        res = sess.run(tri(test_points))
        assert_allclose(res, trinp(test_points))

    def test_projected_evaluate(self, setup):
        """Test evaluations with enabled projection."""
        sess, tri, trinp, test_points = setup

        # Enable project
        trinp.project = True
        tri.project = True

        res = sess.run(tri(test_points))
        assert_allclose(res, trinp(test_points))

    def test_gradient_x(self, setup):
        """Test the gradients with respect to the inputs."""
        sess, tri, trinp, test_points = setup

        points = tf.placeholder(tf.float64, [None, None])
        feed_dict = {points: test_points}

        # Dsiable project
        trinp.project = False
        tri.project = False

        # Just another run test
        y = tri(points)
        res = sess.run(y, feed_dict=feed_dict)
        assert_allclose(res, trinp(test_points))

        # Test gradients
        grad = tf.gradients(y, points)
        res = sess.run(grad, feed_dict=feed_dict)[0]
        assert_allclose(res, trinp.gradient(test_points))

        # Enable project
        trinp.project = True
        tri.project = True

        # Results are different outside of the projection.
        inside = (np.all(test_points < trinp.limits[:, [1]].T, axis=1)
                  & np.all(test_points > trinp.limits[:, [0]].T, axis=1))

        test_points = test_points[inside]

        # Test gradients projected
        y = tri(points)
        grad = tf.gradients(y, points)
        res = sess.run(grad, feed_dict=feed_dict)[0]
        assert_allclose(res[inside], trinp.gradient(test_points))

    def test_gradient_param(self, setup):
        """Test the gradients with respect to the parameters."""
        sess, tri, trinp, test_points = setup

        # Disable project
        trinp.project = True
        tri.project = True

        x = tf.placeholder(tf.float64, [1, 2])

        true_gradient = trinp.parameter_derivative(test_points)
        true_gradient = np.array(true_gradient.todense())

        y = tri(x)
        grad_tf = tf.gradients(y, tri.parameters)[0]
        dense_gradient = np.zeros(true_gradient[0].shape, dtype=np.float)

        for i, test in enumerate(test_points):
            gradient = sess.run(grad_tf, feed_dict={x: test[None, :]})
            dense_gradient[:] = 0.
            dense_gradient[gradient.indices] = gradient.values[:, 0]
            assert_allclose(dense_gradient, true_gradient[i])


def test_neural_network():
    """Test the NeuralNetwork class init."""
    relu = tf.nn.relu

    with tf.Session() as sess:
        nn = NeuralNetwork(layers=[2, 3, 1],
                           nonlinearities=[relu, relu, None])

        # x = tf.placeholder()
        res = nn(np.random.rand(4, 2))
        sess.run(tf.global_variables_initializer())
        res, lipschitz = sess.run([res, nn.lipschitz()])

    assert lipschitz > 0.


if __name__ == '__main__':
    pytest.main()
