"""
msalmlsdm
"""
import numpy as np
from casadi import *
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam


class Constraints:
    """
    Generate neural network, which describes the function ci(s,a)
    """
    def __init__(self, path):
        self.path = path
        self.model = self.generate_model()
        self.weights = []
        self.config = []
        for layer in self.model.layers:
            self.weights.append(layer.get_weights())
            self.config.append(layer.get_config())

    def nn_casadi(self, weights, config, ann_in):
        """
        
        :param weights:
        :param config:
        :param ann_in:
        :return:
        """
        for i in range(len(weights)):
            if 'dense' in config[i]['name']:
                # bias muss "reshaped" werden um konsistent zu sein.
                if i == len(weights) - 1:
                    ann_in = mtimes(ann_in, weights[i][0]) + weights[i][1].reshape(1, np.size(weights[i][0], axis=1))
                else:
                    ann_in = mtimes(ann_in, weights[i][0]) + weights[i][1].reshape(1, np.size(weights[i][0], axis=1))
            if 'activation' in config[i] and 'relu' in config[i]['activation']:
                ann_in = fmax(ann_in, 0)
            elif 'activation' in config[i] and 'tanh' in config[i]['activation']:
                ann_in = tanh(ann_in)
            elif 'activation' in config[i] and 'linear' in config[i]['activation']:
                ann_in = ann_in
        return ann_in

    def generate_model(self):
        """

        :return:
        """
        IN = Input(shape=(3,))
        s1 = Dense(10, activation='tanh')(IN)
        h1 = Dense(10, activation='tanh')(s1)
        Out = Dense(1, activation='linear')(h1)
        model = Model(inputs=IN, outputs=Out)
        adam = Adam(lr=0.001, epsilon=1e-05)
        model.compile(loss='mse', optimizer=adam)
        model.load_weights(self.path)
        return model
