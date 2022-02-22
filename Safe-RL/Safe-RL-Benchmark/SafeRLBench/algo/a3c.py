"""Asynchronous Actor-Critic Agents.

Implementations refer to Denny Britz implementations at
https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c
"""

import copy
import threading

import numpy as np

from SafeRLBench import AlgorithmBase

from SafeRLBench.error import add_dependency

try:
    import tensorflow as tf
    from tensorflow.contrib.distributions import Normal
except ModuleNotFoundError:
    tf = None

import logging

logger = logging.getLogger(__name__)


def _run_thread(alg, worker, sess, coord):
    while not coord.should_stop() and not alg.is_finished():
        p_net_loss, v_net_loss = worker.run(sess)
        alg.step()

    coord.request_stop()


class A3C(AlgorithmBase):
    """Implementation of the Asynchronous Actor-Critic Agents Algorithm.

    Here we use several workers threads, which asynchronously update their
    local estimation networks. The networks consist of a policy estimation
    network which estimates the policy and a value estimation network which
    is used to estimate the expected reward.

    Attributes
    ----------
    environment :
        Environment we want to optimize the policy on.
    policy :
        The policy we want to optimize. The policy has to comply with some
        special requirements to run with this algorithm. It needs to
        implement a tensorflow graph with placeholders stored in the
        attributes X, for the input, and a, for the output.
        Further it will need to implement a ``setup()`` method which will
        effectively assemble the neural network, as well as a ``copy()``
        method that will generate a copy of the network for the worker
        threads. Unfortunately there is no base class for this yet, see
        the ``NeuralNetwork`` policy as a reference implementation.
    max_it : integer
        The maximal number of iterations before we abort.
    num_workers : integer
        Number of workers that should be used asynchronous.
    rate : float
        Rate for Gradient Descent.
    discount : float
        Discount factor for adjusted reward.
    log_file : string
        Indicate the relative path to directory where a summary event file
        should be generated. If `None` no tensorflow logs will be stored.
    done : boolean
        Indicates whether the run is done.
    workers : list
        List containing the worker instances.
    threads : list
        List containing the thread instances

    Notes
    -----
    A proper neural network policy base class does not exist yet, but would
    be very nice to have.

    Bug:
        A3C does not properly run parallel. Actually according to tensorflow
        the GIL should be release upon run invocation, but every thread but
        one just stalls until one worker finished everything on his own.
    """

    def __init__(self, environment, policy, max_it=1000, num_workers=1,
                 rate=0.1, discount=0.1, log_file=False):
        """Initialize A3C.

        Parameters
        ----------
        environment:
            Environment we want to optimize the policy on.
        policy:
            The policy we want to optimize. The policy needs be defined by a
            tensorflow neural network and define certain attributes.
        max_it: int
            Maximum number of iterations.
        num_workers: int
            Number of workers.
        rate: float
            Update rate passed to the optimizer.
        discount: float
            Discount for the computation of the discounted reward.
        log_file: string
            Indicate the relative path to directory where a summary event file
            should be generated. If `None` no tensorflow logs will be stored.
        """
        add_dependency(tf, 'TensorFlow')

        if policy.is_set_up:
            raise(ValueError('Policy should not be set up.'))

        super(A3C, self).__init__(environment, policy, max_it)

        self.num_workers = num_workers
        self.rate = rate
        self.discount = discount

        self.done = False

        self.log_file = log_file

        self.policy = policy

        # init networks
        with tf.device("/cpu:0"):
            with tf.variable_scope('global'):
                self.p_net = _PolicyNet(self.policy, rate)
                self.v_net = _ValueNet(self.policy, rate)

        self.workers = []
        self.threads = []

        self.global_counter = 0

        self.sess = None
        self.coord = None

    def _initialize(self):
        self.global_counter = 0
        self.workers = []
        self.threads = []

        self.done = False

        self.coord = tf.train.Coordinator()

        for i in range(self.num_workers):
            worker = _Worker(self.environment,
                             self.policy,
                             self.p_net,
                             self.v_net,
                             self.discount,
                             'worker_' + str(i))
            self.workers.append(worker)

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        self.policy.sess = self.sess

        # Write a graph file
        if self.log_file:
            graph = self.sess.graph
            writer = tf.summary.FileWriter(self.log_file, graph=graph)
            writer.flush()

    def _step(self):
        self.global_counter += 1

        if self.global_counter % 10 == 0:
            logger.debug("Global update counter at step %d.",
                         self.global_counter)

    def _is_finished(self):
        if self.global_counter >= self.max_it:
            self.done = True
        return self.done

    def _optimize(self):
        self.initialize()
        for worker in self.workers:
            t = threading.Thread(target=_run_thread(self, worker, self.sess,
                                                    self.coord))
            self.threads.append(t)

        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()


class _Worker(object):
    """Worker thread."""

    def __init__(self, env, policy, p_net, v_net, discount, name):
        self.name = name
        self.env = copy.copy(env)
        self.global_policy = policy
        self.global_p_net = p_net
        self.global_v_net = v_net

        self.discount = discount

        # generate local networks
        self.local_policy = policy.copy(name, do_setup=False)
        with tf.variable_scope(name):
            self.local_p_net = _PolicyNet(self.local_policy,
                                          self.global_p_net.rate)
            self.local_v_net = _ValueNet(self.local_policy,
                                         self.global_v_net.rate)

        # create copy op
        trainable_variables = tf.GraphKeys.TRAINABLE_VARIABLES
        self.copy_params_op = self.make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global",
                                          collection=trainable_variables),
            tf.contrib.slim.get_variables(scope=self.name,
                                          collection=trainable_variables))

        # create train ops
        self.p_net_train = self.make_train_op(self.local_p_net,
                                              self.global_p_net)
        self.v_net_train = self.make_train_op(self.local_v_net,
                                              self.global_v_net)

        self.state = self.env.state

    def run(self, sess):
        with sess.as_default():
            sess.run(self.copy_params_op)

            # perform a rollout
            trace = self.env.rollout(self.local_policy)

            advantages = []
            values = []
            states = []
            actions = []

            value = 0.

            tot_reward = 0.
            for (action, state, reward) in trace:
                state = state.flatten()

                value = reward + self.discount * value

                # evaluate value net on state
                value_pred = sess.run(self.local_v_net.V_est,
                                      {self.local_v_net.X: [state]})
                advantage = reward - value_pred.flatten()

                tot_reward += reward

                advantages.append(advantage)
                values.append(value)
                states.append(state)
                actions.append(action)

            print(tot_reward)

            # compute local gradients and train global network
            feed_dict = {
                self.local_p_net.X: np.array(states),
                self.local_p_net.target: advantages,
                self.local_p_net.a: actions,
                self.local_v_net.X: np.array(states),
                self.local_v_net.V: values
            }

            p_net_loss, v_net_loss, _, _ = sess.run([
                self.local_p_net.loss,
                self.local_v_net.loss,
                self.p_net_train,
                self.v_net_train
            ], feed_dict)

        return p_net_loss, v_net_loss

    @staticmethod
    def make_copy_params_op(v1_list, v2_list):
        """Create operation to copy parameters.

        Creates an operation that copies parameters from variable in v1_list to
        variables in v2_list.
        The ordering of the variables in the lists must be identical.
        """
        v1_list = list(sorted(v1_list, key=lambda v: v.name))
        v2_list = list(sorted(v2_list, key=lambda v: v.name))

        update_ops = []
        for v1, v2 in zip(v1_list, v2_list):
            op = v2.assign(v1)
            update_ops.append(op)

        return update_ops

    @staticmethod
    def make_train_op(loc, glob):
        """Create operation that applies local gradients to global network."""
        loc_grads, _ = zip(*loc.grads_and_vars)
        loc_grads, _ = tf.clip_by_global_norm(loc_grads, 5.0)
        _, glob_vars = zip(*glob.grads_and_vars)
        loc_grads_glob_vars = list(zip(loc_grads, glob_vars))
        get_global_step = tf.contrib.framework.get_global_step()

        return glob.opt.apply_gradients(loc_grads_glob_vars,
                                        global_step=get_global_step)


class _ValueNet(object):
    """Wrapper for the Value function."""

    def __init__(self, policy, rate, train=True):
        self.rate = rate

        with tf.variable_scope('value_estimator'):
            self.X = tf.placeholder(policy.dtype,
                                    shape=policy.X.shape,
                                    name='X')
            self.V = tf.placeholder(policy.dtype,
                                    shape=[None],
                                    name='V')

            self.W = policy.init_weights((policy.layers[0], 1))

            self.V_est = tf.matmul(self.X, self.W)

            self.losses = tf.squared_difference(self.V_est, self.V)
            self.loss = tf.reduce_sum(self.losses, name='loss')

            if train:
                self.opt = tf.train.RMSPropOptimizer(rate, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.opt.compute_gradients(self.loss)
                self.grads_and_vars = [(g, v) for g, v in self.grads_and_vars
                                       if g is not None]
                self.update = self.opt.apply_gradients(self.grads_and_vars)


class _PolicyNet(object):
    """Wrapper for the Policy function."""

    def __init__(self, policy, rate, train=True):
        self.rate = rate
        self.policy = policy

        with tf.variable_scope('policy_estimator'):
            self.policy.setup()

            self.X = policy.X
            self.a = policy.a
            self.target = tf.placeholder(dtype='float', shape=[None, 1],
                                         name='target')

            self.a_pred = policy.a_pred
            self.var = policy.var

            dist = Normal(self.a_pred, self.var)
            self.log_probs = dist.log_prob(self.a)

            self.losses = self.log_probs * self.target
            self.loss = tf.reduce_sum(self.losses, name='loss')

            if train:
                self.opt = tf.train.RMSPropOptimizer(rate, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.opt.compute_gradients(self.loss)
                self.grads_and_vars = [(g, v) for g, v in self.grads_and_vars
                                       if g is not None]
                self.update = self.opt.apply_gradients(self.grads_and_vars)
