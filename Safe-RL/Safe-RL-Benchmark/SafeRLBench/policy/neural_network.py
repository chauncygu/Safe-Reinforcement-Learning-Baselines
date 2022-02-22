"""Neural Network Policy implementation."""

from SafeRLBench import Policy
from SafeRLBench.error import add_dependency, MultipleCallsException
from SafeRLBench.spaces import RdSpace

import numpy as np
from numpy.random import normal

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None

import logging

logger = logging.getLogger(__name__)


def default_init_weights(shape):
    """Initialize default weights."""
    weights = tf.random_normal(shape, mean=0, stddev=0.1, name='weights')
    return tf.Variable(weights)


class NeuralNetwork(Policy):
    """Fully connected Neural Network Policy.

    Attributes
    ----------
    args : list
        Contains the args used to initialize the policy.
    kwargs : dict
        Contains the kwargs used to initialize the policy.
    layers : list of integers
        A list describing the layer sizes. The first element represents the
        size of the input layer, the last element the size of the output
        layer.
    state_space : space instance
    action_space : space instance
    weights : tf.Variable
        If none the init_weights function will be used to initialize the
        weights.
    init_weights : callable
        Takes a shape as an argument and returns a tf.Variable according to
        this shape.
    activation : list of activation functions
        An activation function which will be used to construct the respective
        layer. If only one activation function is passed, it will be used for
        every layer. If the argument is None by default the sigmoid function
        will be used.
    dtype : string
        Data type of input and output.
    W_action : list of tf.Variable
        The list contains the `tf.Variable` instances describing the mapping
        between the hidden layers. The i-th entry describes the connection
        between layer i and layer i+1.
    W_var : list of tf.Variable
        This list contains the weights used to compute the variance estimation.
        Each entry corresponds to one layer and contains weights of shape
        (layer[i], 1).
    a_pred :
        Action estimate of the fully connected neural network defined by
        `W_action` and activation.
    var :
        Variance estimate which is a weighted sum of all hidden units.
        The weights are described by `W_var`.
    h : list of tf.Tensor
        Hidden layers
    """

    def __init__(self,
                 layers, weights=None, init_weights=None, activation=None,
                 dtype='float', scope='global', do_setup=False):
        """Initialize Neural Network wrapper."""
        add_dependency(tf, 'TensorFlow')

        if (len(layers) < 2):
            raise ValueError('At least two layers needed.')

        # determine state and action space
        state_space = RdSpace((layers[0],))
        action_space = RdSpace((layers[-1],))

        # store arguments convenient for copy operation
        self.args = [layers]
        self.kwargs = {
            'weights': weights,
            'init_weights': init_weights,
            'activation': activation,
            'dtype': dtype
        }

        self.state_space = state_space
        self.action_space = action_space

        self.dtype = dtype
        self.layers = layers
        self.scope = scope

        self.is_set_up = False

        if init_weights is None:
            self.init_weights = default_init_weights
        else:
            self.init_weights = init_weights

        # Activation function
        if activation is None:
            activation = (len(layers) - 2) * [tf.sigmoid]
        elif (isinstance(activation, list)
                and (len(activation) != len(layers) - 2)):
            raise ValueError('Activation list has wrong size.')
        else:
            activation = (len(layers) - 2) * [activation]

        self.activation = activation

        # Symbols
        self.X = tf.placeholder(dtype, shape=[None, layers[0]], name='X')
        self.a = tf.placeholder(dtype, shape=[None, layers[-1]], name='a')

        if do_setup:
            with tf.variable_scope(self.scope):
                self.setup()
        else:
            # Make sure all fields exist
            self.W_action = None
            self.W_var = None
            self.a_pred = None
            self.var = None
            self.h = None

        self.sess = None

    def setup(self):
        """Set up the network graph.

        The weights and graph will be initialized by this function. If do_setup
        is True, setup will automatically be called, when instantiating the
        class.
        """
        if self.is_set_up:
            raise MultipleCallsException('Network is already set up.')

        layers = self.layers
        weights = self.kwargs['weights']

        # Weights for the action estimation
        with tf.variable_scope('action_estimator'):
            if weights is None:
                w = []
                for i in range(len(layers) - 1):
                    w.append(self.init_weights((layers[i], layers[i + 1])))
            else:
                w = weights

            self.W_action = w

        # generate network
        self.a_pred = self._generate_network()

        # Weights for variance estimation
        with tf.variable_scope('variance_estimator'):
            self.W_var = []
            for i in range(1, len(layers) - 1):
                self.W_var.append(self.init_weights((layers[i], 1)))

        # generate variance network
        self.var = self._generate_variance()

        self.is_set_up = True

    def _generate_network(self):
        self.h = [self.X]
        for i, act in enumerate(self.activation):
            h_i = self.h[i]
            w_i = self.W_action[i]
            self.h.append(act(tf.matmul(h_i, w_i)))

        return tf.matmul(self.h[-1], self.W_action[-1])

    def _generate_variance(self):
        var = []
        if not self.W_var:
            return tf.constant(0, name='variance')
        for h_i, w_i in zip(self.W_var, self.h[1:]):
            var.append(tf.reduce_sum(tf.matmul(w_i, h_i)))
        return tf.abs(tf.reduce_sum(var, name='variance'))

    def copy(self, scope, do_setup=True):
        """Generate a copy of the network.

        The copy will instantiate the class with the same arguments, but
        replace `scope` and `do_setup` with the respective arguments passed
        to this function.

        Parameters
        ----------
        scope : String
            Indication the scope that should be used when initializing the
            network.
        do_setup : Boolean
            Default: True ; Indicating if the `setup` method, should be called
            when instantiating.
        """
        self.kwargs['scope'] = scope
        self.kwargs['do_setup'] = do_setup
        return NeuralNetwork(*self.args, **self.kwargs)

    def map(self, state):
        """Compute output in session.

        Make sure a default session is set when calling.
        """
        state = state.flatten()
        assert(self.state_space.contains(state))

        if self.sess is None:
            sess = tf.get_default_session()
        else:
            sess = self.sess
        mean, var = sess.run([self.a_pred, self.var], {self.X: [state]})

        action = np.array(normal(mean, var))
        action = action.reshape(self.action_space.shape)

        return action

    @property
    def parameters(self):
        """Return weights of the neural network.

        This returns a list of tf.Variables. Please note that these can not
        simply be updated by assignment. See the parameters.setter docstring
        for more information.
        The list of tf.Variables can be directly accessed through the
        attribute `W`.
        """
        if self.sess is None:
            return tf.get_default_session().run(self.W_action + self.W_var)
        else:
            return self.sess.run(self.W_action + self.W_var)

    @parameters.setter
    def parameters(self, update):
        """Setter function for parameters.

        Since the parameters are a list of `tf.Variable`, we need to feed them
        into an assign operator. Thus the argument, needs to be a list
        containing an element for each Variable in `W_action` and `W_var` in
        that order, i.e. `W_var` will be the last element.

        Parameters
        ----------
        update :
            List of parameters for each `tf.Varible`

        Notes
        -----
        Make sure there is a default session or `self.sess` is set.
        """
        if not isinstance(update, list):
            update = [update]

        variables = self.W_action + self.W_var
        assign_op = []

        for (var, val) in zip(variables, update):
            assign_op.append(var.assign(val))

        if self.sess is None:
            sess = tf.get_default_session()
        else:
            sess = self.sess

        sess.run(assign_op)

    @property
    def parameter_space(self):
        """Return parameter space."""
        pass
