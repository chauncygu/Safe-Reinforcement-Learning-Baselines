from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
import tensorflow as tf
import keras.backend as K


class ActorModel:
    """
    Class ActorModel: generates e neural network as actor for the DDPG algorithm
    """
    def __init__(self, sess, nb_observations, nb_actions, nb_disturbance):
        """
        Initialization of neural network
        :param sess:
        :param nb_observations:
        :param nb_actions:
        :param nb_disturbance:
        """

        self.LearningRate = 0.0001
        self.sess = sess
        K.set_session(sess)
        self.nb_obs = nb_observations
        self.nb_actions = nb_actions
        self.nb_disturbance = nb_disturbance
        
        self.model, self.InputS, self.Output, self.weights = self.createActor()
        self.target_model,_, _, _ = self.createActor()
        
        self.action_gradient = tf.placeholder(tf.float32,[None, nb_actions])
        self.params_grad = tf.gradients(self.Output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(self.LearningRate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def createActor(self):
        """
        Function to creat neural network
        :return:
        """

        S = Input(shape=(self.nb_obs+self.nb_disturbance,))

        h0 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(S)  # , kernel_regularizer=regularizers.l1(0.01)

        h1 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(h0)

        h2 = Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01))(h1)

        Out = Dense(self.nb_actions, activation='tanh')(h2)  #tanhh!!!
        actor = Model(inputs=S, outputs=Out)
        return actor, S, Out, actor.trainable_weights

    def train_model(self, states, action_grads):
        """
        Function to train neural network
        :param states:
        :param action_grads:
        :return:
        """
        self.sess.run(self.optimize, feed_dict={
            self.InputS: states,
            self.action_gradient: action_grads
        })
