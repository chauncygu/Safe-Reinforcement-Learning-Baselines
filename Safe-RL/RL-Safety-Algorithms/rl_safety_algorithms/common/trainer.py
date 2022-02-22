import time
from abc import ABC, abstractmethod
import logging
import tensorflow as tf
from tensorflow.python import keras
import rl_safety_algorithms.common.utils as utils


class Trainer(ABC):
    """ abstract base class of the trainer function"""

    def __init__(self,
                 net=None,
                 opt=None,
                 method=None,
                 mode=None,
                 logger=None,
                 log_dir=None,
                 debug_level=None,
                 from_config=None,
                 hooks=None,
                 *args,
                 **kwargs):

        if from_config:
            self.config = from_config
            self.net = from_config.net
            self.opt = from_config.opt
            self.method = from_config.method
            self.mode = from_config.mode
            self.logger = from_config.logger
            self.log_dir = from_config.log_dir
        else:
            self.net = net
            self.opt = opt
            self.method = method
            self.mode = mode
            self.logger = logger
            self.log_dir = log_dir
        self.debug_level = debug_level
        self.training = False
        self.hooks = hooks

        # check if all variables were defined
        assert from_config or (
                    net and opt), 'Provide values for initialization.'
        assert self.log_dir, 'No log path was defined'
        assert self.net, 'No argument parsed for net'
        assert self.opt, 'No argument parsed for opt'
        if logger is None:
            logger = logging.getLogger(__name__)
            logger.warning('There has been no logger defined')

        self.args = args
        self.kwargs = kwargs

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              optimizer=self.opt,
                                              net=self.net)
        self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                  self.log_dir,
                                                  max_to_keep=5)

    def __call__(self, x, *args, **kwargs):
        return self.predict(x)

    def predict(self, x):
        """  predict input according to network
        :param x: np.array or tf.tensor, input data
        :return: tf.Tensor(), holding the prediction of the input x
        """
        return self.net(x)

    def print_params(self):
        print('<', '=' * 20, '>')
        print('log_dir:', self.log_dir)
        print('Network:', self.net)
        print('<', '=' * 20, '>')

    def restore(self):
        """
        restore model from path
        :return: bool, true if restore is successful
        """
        restore_successful = False

        self.checkpoint.restore(
            self.manager.latest_checkpoint)  # .expect_partial()

        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            restore_successful = True
        else:
            print("Initializing parameters from scratch.")
        return restore_successful

    def save(self):
        if self.training:
            utils.mkdir(self.log_dir)
            self.checkpoint.step.assign_add(1)
            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(
                int(self.checkpoint.step), save_path))
        else:
            raise ValueError(
                'Save method is not available, switch to mode = `train`')

    @abstractmethod
    def train(self, total_steps):
        pass


class SupervisedTrainer(Trainer):
    def __init__(self,
                 loss_func,
                 metric,
                 dataset,
                 **kwargs):
        super(SupervisedTrainer, self).__init__(**kwargs)

        self.dataset = dataset
        self.loss_func = loss_func
        self.metric = metric

        self.loss_metric = keras.metrics.Mean(name='test_loss')
        self.acc = keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        self.restore()  # try to restore old checkpoints

    def evaluate_test_set(self):
        losses = []
        for n_batch, batch in enumerate(self.dataset.test):
            data, label = batch
            y = self.net(data, training=False)
            loss = self.loss_func(label, y)
            # mse = tf.reduce_mean(tf.reduce_sum(tf.square(y - label), axis=1)).numpy()
            losses.append(loss.numpy())
        return utils.safe_mean(losses)

    def train(self, total_steps):
        for step in range(total_steps):
            batch_losses = []
            batch_accs = []
            time_start = time.time()
            for i, batch in enumerate(self.dataset.train):
                updates, loss = self.method.get_updates_and_loss(batch,
                                                                 self.net)
                loss_value = self.metric(loss).numpy()
                batch_losses.append(loss_value)
                self.opt.apply_gradients(
                    zip(updates, self.net.trainable_variables))

            write_dic = dict(loss_train=utils.safe_mean(batch_losses),
                             loss_test=self.evaluate_test_set(),
                             time=time.time() - time_start)
            if self.logger:
                self.logger.write(write_dic, step)
            if self.hooks:
                [h.hook() for h in self.hooks]  # call all hooks

        if self.hooks:
            [h.final() for h in self.hooks]  # call all final hook methods

    @tf.function
    def train_step(self, batch):
        image, label = batch
        with tf.GradientTape() as tape:
            predictions = self.net(image, training=True)
            loss = self.loss_func(label, predictions)
        gradients = tape.gradient(loss, self.net.trainable_variables)

        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))

        self.loss_metric(loss)
        self.acc(label, predictions)

        return self.loss_metric.result(), self.acc.result() * 100


class UnsupervisedTrainer(Trainer):
    def __init__(self,
                 loss_func,
                 metric,
                 dataset,
                 *args,
                 **kwargs):
        super(UnsupervisedTrainer, self).__init__(*args, **kwargs)

        self.dataset = dataset
        self.loss_func = loss_func
        self.metric = metric

        self.loss_metric = keras.metrics.Mean(name='test_loss')
        self.acc = keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        self.restore()  # try to restore old checkpoints

    def evaluate_test_set(self):
        """ no tests set to eval """
        pass

    def train(self, total_steps):
        for step in range(total_steps):
            batch_losses = []
            time_start = time.time()
            for i, batch in enumerate(self.dataset.train):
                updates, loss = self.method.get_updates_and_loss(batch,
                                                                 self.net)
                loss_value = self.metric(loss).numpy()
                batch_losses.append(loss_value)
                self.opt.apply_gradients(
                    zip(updates, self.net.trainable_variables))

            write_dic = dict(loss_train=utils.safe_mean(batch_losses),
                             time=time.time() - time_start)
            if self.logger:
                self.logger.write(write_dic, step)
            if self.hooks:
                [h.hook() for h in self.hooks]  # call all hooks

        if self.hooks:
            [h.final() for h in self.hooks]  # call all final hook methods

    @tf.function
    def train_step(self, batch):
        image, label = batch
        with tf.GradientTape() as tape:
            predictions = self.net(image, training=True)
            loss = self.loss_func(label, predictions)
        gradients = tape.gradient(loss, self.net.trainable_variables)

        self.opt.apply_gradients(zip(gradients, self.net.trainable_variables))

        self.loss_metric(loss)
        self.acc(label, predictions)

        return self.loss_metric.result(), self.acc.result() * 100


class ReinforcementTrainer(Trainer, ABC):
    def __init__(self,
                 env,
                 *args,
                 **kwargs):
        super(ReinforcementTrainer, self).__init__(*args,
                                                   **kwargs)
        self.env = env
        self.print_params()

    def print_params(self):
        super(ReinforcementTrainer, self).print_params()
        print('env:', self.env)
