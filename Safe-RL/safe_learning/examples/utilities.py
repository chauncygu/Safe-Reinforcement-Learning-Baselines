from __future__ import division, print_function

import sys
import os
import importlib
import numpy as np
import scipy
import tensorflow as tf
from scipy import signal
from matplotlib.colors import ListedColormap

from safe_learning import config, DeterministicFunction, GridWorld
from safe_learning.utilities import concatenate_inputs
if sys.version_info.major == 2:
    import imp

__all__ = ['compute_roa', 'binary_cmap', 'constrained_batch_sampler', 'compute_closedloop_response',
           'get_parameter_change', 'import_from_directory', 'LyapunovNetwork', 'InvertedPendulum', 'CartPole']

NP_DTYPE = config.np_dtype
TF_DTYPE = config.dtype

def import_from_directory(library, path):
    """Import a library from a directory outside the path.

    Parameters
    ----------
    library: string
        The name of the library.
    path: string
        The path of the folder containing the library.

    """
    try:
        return importlib.import_module(library)
    except ImportError:
        module_path = os.path.abspath(path)
        version = sys.version_info

        if version.major == 2:
            f, filename, desc = imp.find_module(library, [module_path])
            return imp.load_module(library, f, filename, desc)
        else:
            sys.path.append(module_path)
            return importlib.import_module(library)


class LyapunovNetwork(DeterministicFunction):
    """A positive-definite neural network."""

    def __init__(self, input_dim, layer_dims, activations, eps=1e-6,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 name='lyapunov_network'):
        super(LyapunovNetwork, self).__init__(name=name)
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        self.activations = activations
        self.eps = eps
        self.initializer = initializer

        if layer_dims[0] < input_dim:
            raise ValueError('The first layer dimension must be at least the input dimension!')

        if np.all(np.diff(layer_dims) >= 0):
            self.output_dims = layer_dims
        else:
            raise ValueError('Each layer must maintain or increase the dimension of its input!')

        self.hidden_dims = np.zeros(self.num_layers, dtype=int)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            self.hidden_dims[i] = np.ceil((layer_input_dim + 1) / 2).astype(int)

        # For printing results nicely
        self.layer_partitions = np.zeros(self.num_layers, dtype=int)
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            if dim_diff > 0:
                self.layer_partitions[i] = 2
            else:
                self.layer_partitions[i] = 1

    def build_evaluation(self, points):
        """Build the evaluation graph."""
        net = points
        if isinstance(net, np.ndarray):
            net = tf.constant(net)
        for i in range(self.num_layers):
            if i == 0:
                layer_input_dim = self.input_dim
            else:
                layer_input_dim = self.output_dims[i - 1]
            W = tf.get_variable('weights_posdef_{}'.format(i), [self.hidden_dims[i], layer_input_dim], TF_DTYPE, self.initializer)
            kernel = tf.matmul(W, W, transpose_a=True) + self.eps * tf.eye(layer_input_dim, dtype=TF_DTYPE)
            dim_diff = self.output_dims[i] - layer_input_dim
            if dim_diff > 0:
                W = tf.get_variable('weights_{}'.format(i), [dim_diff, layer_input_dim], TF_DTYPE, self.initializer)
                kernel = tf.concat([kernel, W], axis=0)
            layer_output = tf.matmul(net, kernel, transpose_b=True)
            net = self.activations[i](layer_output, name='layer_output_{}'.format(i))
        values = tf.reduce_sum(tf.square(net), axis=1, keepdims=True, name='quadratic_form')
        return values

    def print_params(self):
        offset = 0
        params = self.parameters.eval()
        for i, dim_diff in enumerate(np.diff(np.concatenate([[self.input_dim], self.output_dims]))):
            print('Layer weights {}:'.format(i))
            W0 = params[offset + i]
            print('W0:\n{}'.format(W0))
            if dim_diff > 0:
                W1 = params[offset + 1 + i]
                print('W1:\n{}'.format(W1))
            else:
                offset += 1
            kernel = W0.T.dot(W0) + self.eps * np.eye(W0.shape[1])
            eigvals, _ = np.linalg.eig(kernel)
            print('Eigenvalues of (W0.T*W0 + eps*I):', eigvals, '\n')


class RBFNetwork(DeterministicFunction):
    def __init__(self, limits, num_states, variances=None, initializer=tf.contrib.layers.xavier_initializer(), name='rbf_network'):
        super(RBFNetwork, self).__init__(name=name)
        self.discretization = GridWorld(limits, num_states)
        if variances is not None:
            self.variances = variances
        else:
            self.variances = np.min(self.discretization.unit_maxes) ** 2
        self._initializer = initializer
        self._betas = 1 / (2 * self.variances)
        self.centres = self.discretization.all_points
        self._centres_3D = np.reshape(self.centres.T, (1, self.discretization.ndim, self._hidden_units))

    def build_evaluation(self, states):
        W = tf.get_variable('weights', dtype=TF_DTYPE, shape=[self.discretization.nindex, 1], initializer=self._initializer)
        states_3D = tf.expand_dims(states, axis=2)
        phi_X = tf.exp(-self._betas * tf.reduce_sum(tf.square(states_3D - self._centres_3D), axis=1, keep_dims=False))
        output = tf.matmul(phi_X, W)
        return output


class InvertedPendulum(DeterministicFunction):
    """Inverted Pendulum.

    Parameters
    ----------
    mass : float
    length : float
    friction : float, optional
    dt : float, optional
        The sampling time.
    normalization : tuple, optional
        A tuple (Tx, Tu) of arrays used to normalize the state and actions. It
        is so that diag(Tx) *x_norm = x and diag(Tu) * u_norm = u.

    """

    def __init__(self, mass, length, friction=0, dt=1 / 80,
                 normalization=None):
        """Initialization; see `InvertedPendulum`."""
        super(InvertedPendulum, self).__init__()
        self.mass = mass
        self.length = length
        self.gravity = 9.81
        self.friction = friction
        self.dt = dt

        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    @property
    def inertia(self):
        """Return inertia of the pendulum."""
        return self.mass * self.length ** 2

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = tf.matmul(state, Tx_inv)

        if action is not None:
            action = tf.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)

        state = tf.matmul(state, Tx)
        if action is not None:
            action = tf.matmul(action, Tu)

        return state, action

    def linearize(self):
        """Return the linearized system.

        Returns
        -------
        a : ndarray
            The state matrix.
        b : ndarray
            The action matrix.

        """
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        A = np.array([[0, 1],
                      [gravity / length, -friction / inertia]],
                     dtype=config.np_dtype)

        B = np.array([[0],
                      [1 / inertia]],
                     dtype=config.np_dtype)

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)

            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        sys = signal.StateSpace(A, B, np.eye(2), np.zeros((2, 1)))
        sysd = sys.to_discrete(self.dt)
        return sysd.A, sysd.B

    @concatenate_inputs(start=1)
    def build_evaluation(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = tf.split(state_action, [2, 1], axis=1)
        state, action = self.denormalize(state, action)

        n_inner = 10
        dt = self.dt / n_inner
        for i in range(n_inner):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        # Physical dynamics
        gravity = self.gravity
        length = self.length
        friction = self.friction
        inertia = self.inertia

        angle, angular_velocity = tf.split(state, 2, axis=1)

        x_ddot = gravity / length * tf.sin(angle) + action / inertia

        if friction > 0:
            x_ddot -= friction / inertia * angular_velocity

        state_derivative = tf.concat((angular_velocity, x_ddot), axis=1)

        # Normalize
        return state_derivative


class CartPole(DeterministicFunction):
    """Cart with mounted inverted pendulum.

    Parameters
    ----------
    pendulum_mass : float
    cart_mass : float
    length : float
    dt : float, optional
        The sampling period used for discretization.
    normalization : tuple, optional
        A tuple (Tx, Tu) of 1-D arrays or lists used to normalize the state and
        action, such that x = diag(Tx) * x_norm and u = diag(Tu) * u_norm.

    """

    def __init__(self, pendulum_mass, cart_mass, length, rot_friction=0.0,
                 dt=0.01, normalization=None):
        """Initialization; see `CartPole`."""
        super(CartPole, self).__init__(name='CartPole')
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.rot_friction = rot_friction
        self.dt = dt
        self.gravity = 9.81
        self.state_dim = 4
        self.action_dim = 1
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = tf.matmul(state, Tx_inv)

        if action is not None:
            action = tf.matmul(action, Tu_inv)

        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)

        state = tf.matmul(state, Tx)
        if action is not None:
            action = tf.matmul(action, Tu)

        return state, action

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.
        Bd : ndarray
            The discrete-time action matrix.

        """
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.rot_friction
        g = self.gravity

        A = np.array([[0, 0,                     1, 0                            ],
                      [0, 0,                     0, 1                            ],
                      [0, g * m / M,             0, -b / (M * L)                 ],
                      [0, g * (m + M) / (L * M), 0, -b * (m + M) / (m * M * L**2)]],
                     dtype=config.np_dtype)

        B = np.array([0, 0, 1 / M, 1 / (M * L)]).reshape((-1, self.action_dim))

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        Ad, Bd, _, _, _ = signal.cont2discrete((A, B, 0, 0), self.dt,
                                               method='zoh')
        return Ad, Bd

    @concatenate_inputs(start=1)
    def build_evaluation(self, state_action):
        """Evaluate the dynamics."""
        # Denormalize
        state, action = tf.split(state_action, [4, 1], axis=1)
        state, action = self.denormalize(state, action)

        inner_euler_steps = 10
        dt = self.dt / inner_euler_steps
        for _ in range(inner_euler_steps):
            state_derivative = self.ode(state, action)
            state = state + dt * state_derivative

        return self.normalize(state, None)[0]

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.
        action: ndarray or Tensor
            Actions.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        # Physical dynamics
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.rot_friction
        g = self.gravity

        x, theta, v, omega = tf.split(state, [1, 1, 1, 1], axis=1)

        x_dot = v
        theta_dot = omega

        det = L*(M + m*tf.square(tf.sin(theta)))
        v_dot = (action - m*L*tf.square(omega)*tf.sin(theta) - b*omega*tf.cos(theta) + 0.5*m*g*L*tf.sin(2*theta)) * L/det
        omega_dot = (action*tf.cos(theta) - 0.5*m*L*tf.square(omega)*tf.sin(2*theta) - b*(m + M)*omega/(m*L)
                     + (m + M)*g*tf.sin(theta)) / det

        state_derivative = tf.concat((x_dot, theta_dot, v_dot, omega_dot), axis=1)

        return state_derivative


class VanDerPol(DeterministicFunction):
    """Van der Pol oscillator in reverse-time."""

    def __init__(self, damping=1, dt=0.01, normalization=None):
        """Initialization; see `VanDerPol`."""
        super(VanDerPol, self).__init__(name='VanDerPol')
        self.damping = damping
        self.dt = dt
        self.state_dim = 2
        self.action_dim = 0
        self.normalization = normalization
        if normalization is not None:
            self.normalization = np.array(normalization, dtype=config.np_dtype)
            self.inv_norm = self.normalization ** -1

    def normalize(self, state):
        """Normalize states and actions."""
        if self.normalization is None:
            return state
        Tx_inv = np.diag(self.inv_norm)
        state = tf.matmul(state, Tx_inv)
        return state

    def denormalize(self, state):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state
        Tx = np.diag(self.normalization)
        state = tf.matmul(state, Tx)
        return state

    def linearize(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.

        """
        A = np.array([[0, -1], [1, -1]], dtype=config.np_dtype)
        if self.normalization is not None:
            Tx = np.diag(self.normalization)
            Tx_inv = np.diag(self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
        B = np.zeros([2, 1])
        Ad, _, _, _, _ = signal.cont2discrete((A, B, 0, 0), self.dt, method='zoh')
        return Ad

    @concatenate_inputs(start=1)
    def build_evaluation(self, state_action):
        """Evaluate the dynamics."""
        state, _ = tf.split(state_action, [2, 1], axis=1)
        state = self.denormalize(state)
        inner_euler_steps = 10
        dt = self.dt / inner_euler_steps
        for _ in range(inner_euler_steps):
            state_derivative = self.ode(state)
            state = state + dt * state_derivative
        return self.normalize(state)

    def ode(self, state):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        x, y = tf.split(state, 2, axis=1)
        x_dot = - y
        y_dot = x + self.damping * (x ** 2 - 1) * y
        state_derivative = tf.concat((x_dot, y_dot), axis=1)
        return state_derivative


def reward_rollout(grid, closed_loop_dynamics, reward_function, discount, horizon=250, tol=1e-3):
    """Compute a finite-horizon rollout of a reward function over a discretization of the state space."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex

    converged = False
    rollout = np.zeros(nindex)
    current_states = all_points
    for t in range(horizon):
        temp = (discount ** t) * reward_function(current_states).ravel()
        rollout += temp
        if np.max(np.abs(temp)) < tol:
            converged = True
            break
        current_states = closed_loop_dynamics(current_states)
    if converged:
        print('Reward sums converged after {} steps!'.format(t + 1))
    else:
        print('Reward sums did not converge!')
    return rollout


def constrained_batch_sampler(dynamics, policy, state_dim, batch_size, action_limit=None, zero_pad=0):
    """Sample states that do not map outside a bounded state space or to saturated control inputs."""
    batch = tf.random_uniform([int(batch_size), state_dim], -1, 1,
                              dtype=TF_DTYPE, name='batch_sample')
    actions = policy(batch)
    future_batch = dynamics(batch, actions)
    maps_inside = tf.reduce_all(tf.logical_and(future_batch >= -1, future_batch <= 1), axis=1)
    maps_inside_idx = tf.squeeze(tf.where(maps_inside))
    constrained_batch = tf.gather(batch, maps_inside_idx)
    if action_limit is not None:
        c = np.abs(action_limit)
        undersaturated = tf.reduce_all(tf.logical_and(actions >= -c, actions <= c), axis=1)
        undersaturated_idx = tf.squeeze(tf.where(undersaturated))
        constrained_batch = tf.gather(batch, undersaturated_idx)
    if constrained_batch.get_shape()[0] == 0:
        constrained_batch = tf.zeros([1, state_dim], dtype=TF_DTYPE)
    if zero_pad > 0:
        zero_padding = tf.constant([[0, int(zero_pad)], [0, 0]])
        constrained_batch = tf.pad(constrained_batch, zero_padding)
    return constrained_batch


def get_parameter_change(old_params, new_params, ord='inf'):
    """Measure the change in parameters.

    Parameters
    ----------
    old_params : list
        The old parameters as a list of ndarrays, typically from
        session.run(var_list)
    new_params : list
        The old parameters as a list of ndarrays, typically from
        session.run(var_list)
    ord : {non-zero int, inf, -inf, ‘fro’, ‘nuc’}, optional
        Type of norm used to quantify the change. Passed to `numpy.linalg.norm`.

    Returns
    -------
    change : float
        The parameter change measured as a norm of the vector difference.

    """
    if ord=='inf':
        ord = np.inf
    elif ord=='-inf':
        ord = -np.inf

    old_params = np.concatenate([param.ravel() for param in old_params])
    new_params = np.concatenate([param.ravel() for param in new_params])
    change = np.linalg.norm(new_params - old_params, ord=ord)

    return change


def compute_closedloop_response(dynamics, policy, state_dim, steps, dt, reference='zero', const=1.0, ic=None):
    """Compute the closed-loop dynamic response to different reference signals."""
    action_dim = policy.output_dim

    if reference == 'impulse':
        r = np.zeros((steps + 1, action_dim))
        r[0, :] = (1 / dt) * np.ones((1, action_dim))
    elif reference == 'step':
        r = const*np.ones((steps + 1, action_dim))
    elif reference == 'zero':
        r = np.zeros((steps + 1, action_dim))

    times = dt*np.arange(steps + 1, dtype=NP_DTYPE).reshape((-1, 1))
    states = np.zeros((steps + 1, state_dim), dtype=NP_DTYPE)
    actions = np.zeros((steps + 1, action_dim), dtype=NP_DTYPE)
    if ic is not None:
        states[0, :] = np.asarray(ic, dtype=NP_DTYPE).reshape((1, state_dim))

    session = tf.get_default_session()
    with tf.name_scope('compute_closedloop_response'):
        current_ref = tf.placeholder(TF_DTYPE, shape=[1, action_dim])
        current_state = tf.placeholder(TF_DTYPE, shape=[1, state_dim])
        current_action = policy(current_state)
        next_state = dynamics(current_state, current_action + current_ref)
        data = [current_action, next_state]

    for i in range(steps):
        feed_dict = {current_ref: r[[i], :], current_state: states[[i], :]}
        actions[i, :], states[i + 1, :] = session.run(data, feed_dict)

    # Get the last action for completeness
    feed_dict = {current_ref: r[[-1], :], current_state: states[[-1], :]}
    actions[-1, :], _ = session.run(data, feed_dict)

    return states, actions, times, r


def gridify(norms, maxes=None, num_points=25):
    """Construct a discretization."""
    norms = np.asarray(norms).ravel()
    if maxes is None:
        maxes = norms
    else:
        maxes = np.asarray(maxes).ravel()
    limits = np.column_stack((- maxes / norms, maxes / norms))

    if isinstance(num_points, int):
        num_points = [num_points, ] * len(norms)
    grid = GridWorld(limits, num_points)
    return grid


def compute_roa(grid, closed_loop_dynamics, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""
    if isinstance(grid, np.ndarray):
        all_points = grid
        nindex = grid.shape[0]
        ndim = grid.shape[1]
    else: # grid is a GridWorld instance
        all_points = grid.all_points
        nindex = grid.nindex
        ndim = grid.ndim

    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states = all_points
        for t in range(1, horizon):
            end_states = closed_loop_dynamics(end_states)
    else:
        trajectories = np.empty((nindex, ndim, horizon))
        trajectories[:, :, 0] = all_points
        for t in range(1, horizon):
            trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
        end_states = trajectories[:, :, -1]

    if equilibrium is None:
        equilibrium = np.zeros((1, ndim))

    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories


def binary_cmap(color='red', alpha=1.):
    """Construct a binary colormap."""
    if color == 'red':
        color_code = (1., 0., 0., alpha)
    elif color == 'green':
        color_code = (0., 1., 0., alpha)
    elif color == 'blue':
        color_code = (0., 0., 1., alpha)
    else:
        color_code = color
    transparent_code = (1., 1., 1., 0.)
    return ListedColormap([transparent_code, color_code])


def find_nearest(array, value, sorted_1d=True):
    """Find the nearest value to a reference value and its index in an array."""
    if not sorted_1d:
        array = np.sort(array)
    idx = np.searchsorted(array, value, side='left')
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx - 1]) < np.abs(value - array[idx])):
        idx -= 1
    return idx, array[idx]


def balanced_confusion_weights(y, y_true, scale_by_total=True):
    """Compute class weights from a confusion matrix."""
    y = y.astype(np.bool)
    y_true = y_true.astype(np.bool)

    # Assuming labels in {0, 1}, count entries from confusion matrix
    TP = (y & y_true).sum()
    TN = (~y & ~y_true).sum()
    FP = (y & ~y_true).sum()
    FN = (~y & y_true).sum()
    confusion_counts = np.array([[TN, FN], [FP, TP]])

    # Scale up each sample by inverse of confusion weight
    weights = np.ones_like(y, dtype=float)
    weights[y & y_true] /= TP
    weights[~y & ~y_true] /= TN
    weights[y & ~y_true] /= FP
    weights[~y & y_true] /= FN
    if scale_by_total:
        weights *= y.size

    return weights, confusion_counts


def balanced_class_weights(y_true, scale_by_total=True):
    """Compute class weights from class label counts."""
    y = y_true.astype(np.bool)
    nP = y.sum()
    nN = y.size - y.sum()
    class_counts = np.array([nN, nP])

    weights = np.ones_like(y, dtype=float)
    weights[ y] /= nP
    weights[~y] /= nN
    if scale_by_total:
        weights *= y.size

    return weights, class_counts


def monomials(x, deg):
    """Compute monomial features of `x' up to degree `deg'."""
    x = np.atleast_2d(np.copy(x))
    # 1-D features (x, y)
    Z = x
    if deg >= 2:
        # 2-D features (x^2, x * y, y^2)
        temp = np.empty([len(x), 3])
        temp[:, 0] = x[:, 0] ** 2
        temp[:, 1] = x[:, 0] * x[:, 1]
        temp[:, 2] = x[:, 1] ** 2
        Z = np.hstack((Z, temp))
    if deg >= 3:
        # 3-D features (x^3, x^2 * y, x * y^2, y^3)
        temp = np.empty([len(x), 4])
        temp[:, 0] = x[:, 0] ** 3
        temp[:, 1] = (x[:, 0] ** 2) * x[:, 1]
        temp[:, 2] = x[:, 0] * (x[:, 1] ** 2)
        temp[:, 3] = x[:, 1] ** 3
        Z = np.hstack((Z, temp))
    if deg >= 4:
        # 4-D features (x^4, x^3 * y, x^2 * y^2, x * y^3, y^4)
        temp = np.empty([len(x), 5])
        temp[:, 0] = x[:, 0] ** 4
        temp[:, 1] = (x[:, 0] ** 3) * x[:, 1]
        temp[:, 2] = (x[:, 0] ** 2) * (x[:, 1] ** 2)
        temp[:, 3] = x[:, 0] * (x[:, 1] ** 3)
        temp[:, 4] = x[:, 1] ** 4
        Z = np.hstack((Z, temp))
    return Z


def derivative_monomials(x, deg):
    """Compute derivatives of monomial features of `x' up to degree `deg'."""
    x = np.atleast_2d(np.copy(x))
    dim = x.shape[1]
    # 1-D features (x, y)
    Z = np.zeros([len(x), 2, dim])
    Z[:, 0, 0] = 1
    Z[:, 1, 1] = 1
    if deg >= 2:
        # 2-D features (x^2, x * y, y^2)
        temp = np.zeros([len(x), 3, dim])
        temp[:, 0, 0] = 2 * x[:, 0]
        temp[:, 1, 0] = x[:, 1]
        temp[:, 1, 1] = x[:, 0]
        temp[:, 2, 1] = 2 * x[:, 1]
        Z = np.concatenate((Z, temp), axis=1)
    if deg >= 3:
        # 3-D features (x^3, x^2 * y, x * y^2, y^3)
        temp = np.zeros([len(x), 4, dim])
        temp[:, 0, 0] = 3 * x[:, 0] ** 2
        temp[:, 1, 0] = 2 * x[:, 0] * x[:, 1]
        temp[:, 1, 1] = x[:, 0] ** 2
        temp[:, 2, 0] = x[:, 1] ** 2
        temp[:, 2, 1] = 2 * x[:, 0] * x[:, 1]
        temp[:, 3, 1] = 3 * x[:, 1] ** 2
        Z = np.concatenate((Z, temp), axis=1)
    return Z
