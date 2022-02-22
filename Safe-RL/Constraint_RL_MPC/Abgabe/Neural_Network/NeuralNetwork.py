"""
Generates neural network object
"""
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adagrad
from casadi import *


class NN:

    def __init__(self, num_in, num_out, num_hidden, activation_in, activation_hidden, activation_out, optimizer,
                 path=None):
        """

        :param num_in: input dimension
        :param num_out:  output dimension
        :param num_hidden: number of neurons for hidden layer
        :param activation_in: activation function for input layer
        :param activation_hidden: activation function for hidden layer
        :param activation_out: activation function for output layer
        :param optimizer: defines the optimizer to be used
        :param path: set if weights are already trained
        """

        self.num_in = num_in
        self.num_out = num_out
        self.num_hidden = num_hidden

        self.activation_in = activation_in
        self.activation_hidden = activation_hidden
        self.activation_out = activation_out

        self.optimizer = optimizer

        # generate network
        self.model = self.generate_NN()

        # load weights
        if path is not None:
            self.model.load_weights(path)

        # save weights and configuration for each layer
        self.weights = []
        self.config = []
        self.get_conf_weights()

    def generate_NN(self):
        """
        Function to generate the structure of a neural network
        :return:
        """
        input = Input(shape=(self.num_in,))
        s1 = Dense(self.num_hidden, activation=self.activation_in)(input)
        h1 = Dense(self.num_hidden, activation=self.activation_hidden)(s1)
        output = Dense(self.num_out, activation=self.activation_out)(h1)
        model = Model(inputs=input, outputs=output)
        if self.optimizer is 'Adam':
            optimizer = Adam(lr=0.001, epsilon=1e-05)
        elif self.optimizer is 'Adagrad':
            optimizer = Adagrad(lr=0.001, epsilon=1e-05)

        model.compile(loss='mse', optimizer=optimizer)
        return model

    def nn_casadi(self, weights, config, ann_in):
        """
        Function to convert NN from Keras framework in casadi framework
        :param weights: all weights per layer
        :param config: configuration per layer
        :param ann_in: input
        :return:
        """

        for i in range(len(weights)):
            if 'dense' in config[i]['name']:
                ann_in = mtimes(ann_in, weights[i][0]) + weights[i][1].reshape(1, np.size(weights[i][0], axis=1))
            if 'activation' in config[i] and 'relu' in config[i]['activation']:
                ann_in = fmax(ann_in, 0)
            elif 'activation' in config[i] and 'tanh' in config[i]['activation']:
                ann_in = tanh(ann_in)
            elif 'activation' in config[i] and 'linear' in config[i]['activation']:
                ann_in = ann_in
        return ann_in

    def get_conf_weights(self):
        """
        Function saves NN weights and configuration seperatly
        :return:
        """
        for layer in self.model.layers:
            self.weights.append(layer.get_weights())
            self.config.append(layer.get_config())

        return
