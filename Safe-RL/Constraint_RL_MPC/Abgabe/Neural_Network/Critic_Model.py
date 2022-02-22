import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,merge, Add
from keras.optimizers import Adam, Adagrad
import tensorflow as tf
import keras.backend as K
from keras import regularizers


class CriticModel:
    def __init__(self, sess, nb_observations, nb_actions, env, nb_disturbance):
        self.LearningRate = 0.001
        self.env = env
        self.sess = sess
        K.set_session(sess)
        self.nb_obs = nb_observations
        self.nb_actions = nb_actions
        self.nb_disturbance = nb_disturbance
        
        self.model, self.InputStates, self.InputActions, self.Output = self.createCritic()
        self.target_model, target_model_in, target_model_in2, target_model_out = self.createCritic()

        # define gradients dQ/dA
        self.grads_dQ_da = tf.gradients(self.model.output, self.InputActions)

        self.sess.run(tf.global_variables_initializer())

    def createCritic(self): 
        
        S = Input(shape=(self.nb_obs+self.nb_disturbance,))
        A = Input(shape=(self.nb_actions,))
        s1 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(S)
        a1 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(A)
        s2 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(s1)
        merged = Add()([a1, s2])
        merged_h1 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(merged)
        h3 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(merged_h1)
        V = Dense(1,activation='linear')(h3)
        critic = Model(inputs=[S, A], outputs=V)
        adagrad = Adam(lr=self.LearningRate)  # , epsilon=None, decay=0.0)
        critic.compile(loss='mse', optimizer=adagrad)
        return critic, S, A, V


    def criticGradients(self, states, actions):
        return self.sess.run(self.grads_dQ_da, feed_dict={self.InputStates: states, self.InputActions: actions})[0]






