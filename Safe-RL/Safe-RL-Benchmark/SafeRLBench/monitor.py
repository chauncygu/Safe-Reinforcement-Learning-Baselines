"""Monitoring implementations."""

import logging
import time

from SafeRLBench import config

from contextlib import contextmanager

logger = logging.getLogger(__name__)

__all__ = ('EnvMonitor', 'AlgoMonitor')


class EnvMonitor(object):
    """
    Environment Monitor, providing tracking for environments.

    Attributes
    ----------
    monitor :
        This is the container where monitoring data will be stored.

    Methods
    -------
    monitor_update()
        Context manager for monitoring environment updates. It should be used
        when invoking the private ``_update`` implementation from the interface
        method.
    monitor_rollout()
        Context manager for monitoring environment rollout. It should be used
        when invoking the private ``_rollout`` implementation from the
        interface method.
    monitor_reset()
        Context manager for monitoring environment resets. It should be used
        when invoking the private ``_reset`` implementation from the interface
        method.
    """

    def __new__(cls, *args, **kwargs):
        """Create monitor in subclasses."""
        obj = object.__new__(cls)
        obj.monitor = EnvData()
        return obj

    @contextmanager
    def monitor_update(self):
        """Context monitoring update."""
        self._before_update()
        yield self
        self._after_update()

    @contextmanager
    def monitor_rollout(self):
        """Context monitoring rollout."""
        self._before_rollout()
        yield self
        self._after_rollout()

    @contextmanager
    def monitor_reset(self):
        """Context monitoring reset."""
        self._before_reset()
        yield self
        self._after_reset()

    def _before_update(self):
        """Monitor environment before update.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _after_update(self):
        """Monitor environment after update.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _before_rollout(self):
        """Monitor environment before rollout.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _after_rollout(self):
        """
        Monitor environment after rollout.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        self.monitor.rollout_cnt += 1

    def _before_reset(self):
        """Monitor environment before reset.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass

    def _after_reset(self):
        """Monitor environment after reset.

        Parameters
        ----------
        env :
            Environment instance to be monitored.
        """
        pass


class AlgoMonitor(object):
    """Algorithm monitor tracks algorithms' activity.

    This class is inherited by the `AlgorithmBase` class and will provide it
    with tracking capabilities.

    Attributes
    ----------
    monitor :
        This is the container where monitoring data will be stored.
    grad :
        The Alogrithm can set this field, to provide information about the
        current gradient to the monitor.
    has_policy :
        In case the algorithm does not depend on a policy and does not need
        any parameters, this can me set to False, to prevent issues with
        tracking data that does not exist.

    Methods
    -------
    monitor_optimize()
        Context manager for monitoring algorithm optimizations. It should be
        used when invoking the private ``_optimize`` implementation from the
        interface method.
    monitor_initialize()
        Context manager for monitoring algorithm initializations. It should be
        used when invoking the private ``_initialize`` implementation from the
        interface method.
    monitor_step()
        Context manager for monitoring algorithm step. It should be used when
        invoking the private ``_step`` implementation from the interface
        method.
    """

    def __new__(cls, *args, **kwargs):
        """Create monitor in subclasses."""
        obj = object.__new__(cls)
        obj.monitor = AlgoData()
        obj.grad = None
        obj.has_policy = True
        return obj

    @contextmanager
    def monitor_optimize(self):
        """Context monitoring optimization."""
        self._before_optimize()
        yield self
        self._after_optimize()

    @contextmanager
    def monitor_initialize(self):
        """Context monitoring initialize."""
        yield self
        if self.has_policy:
            self.monitor.parameters.append(self.policy.parameters)

    @contextmanager
    def monitor_step(self):
        """Context monitoring stepping."""
        self._before_step()
        yield self
        self._after_step()

    def _before_optimize(self):
        """Set monitor up for optimization run.

        Parameters
        ----------
        alg :
            the algorithm instance to be monitored
        """
        if config.monitor_verbosity > 0:
            logger.info('Starting optimization of %s...', str(self))

        # reset monitor object in case of rerun
        self.monitor.reset()

        # init monitor dict for algorithm
        self.monitor.t = time.time()

        # init optimization time control
        self.monitor.optimize_start = time.time()

    def _after_optimize(self):
        """Catch data after optimization run."""
        # retrieve time of optimization
        optimize_end = time.time()
        optimize_time = optimize_end - self.monitor.optimize_start

        if self.monitor.optimize_start == 0:
            logger.warning('Time measure for optimize corrupted')

        self.monitor.optimize_start = 0

        self.monitor.optimize_time = optimize_time

        # if the gradient attribute has been set
        if self.grad is not None:
            logger.debug('Finished optimization after %d steps with grad %s.',
                         self.monitor.step_cnt, str(self.grad))
        else:
            logger.debug('Finished optimization after %d steps.',
                         self.monitor.step_cnt)

        if self.has_policy:
            # independently compute traces after optimization is finished
            if config.monitor_verbosity > 0:
                logger.info('Computing traces for %s run...', str(self))

            for parameters in self.monitor.parameters:

                self.policy.parameters = parameters

                # compute trace
                trace = self.environment._rollout(self.policy)
                self.monitor.traces.append(trace)

                # compute total reward
                reward = sum([t[2] for t in trace])
                self.monitor.rewards.append(reward)

    def _before_step(self):
        """Monitor algorithm before step.

        Parameters
        ----------
        alg :
            Algorithm instance to be monitored.
        """
        # count the number of rollouts for each step
        self.environment.monitor.rollout_cnt = 0

        if config.monitor_verbosity > 2:
            logger.info('Computing step %d for %s...', self.monitor.step_cnt,
                        str(self))

    def _after_step(self):
        """Monitor algorithm after step.

        Parameters
        ----------
        alg :
            Algorithm instance to be monitored.
        """
        emonitor = self.environment.monitor

        self.monitor.step_cnt += 1

        # store the number of rollouts
        self.monitor.rollout_cnts.append(emonitor.rollout_cnt)

        # retrieve information from the policy
        if self.has_policy:
            # retrieve current parameters
            parameters = self.policy.parameters
            # store information
            self.monitor.parameters.append(parameters)

        # log if wanted
        self._step_log()

    def _step_log(self):
        # print information if wanted
        monitor = self.monitor
        n = monitor.step_cnt
        log = 0

        # check verbosity level
        if config.monitor_verbosity > 0:
            if monitor.step_cnt % 1000 == 0:
                log = 1000

        if config.monitor_verbosity > 1:
            if monitor.step_cnt % 100 == 0:
                log = 100

        if config.monitor_verbosity > 2:
            log = 1

        if log:
            # generate time strings
            now = time.time()
            t = now - monitor.optimize_start
            t_s = "{:.2f}".format(t)
            avg_s = "{:.3f}".format(t / n)

            # generate log message
            msg = 'Status for ' + self.__class__.__name__ + ' on '
            msg += self.environment.__class__.__name__ + ':\n\n'
            msg += '\tRun: %d\tTime: %s\t Avg: %s\n' % (n, t_s, avg_s)
            if self.has_policy:
                # retrieve current state
                par_s = str(self.policy.parameters)
                msg += '\tParameter: \t%s\n' % (par_s)

            logger.info(msg)

    def _alg_reset(self):
        """Reset the algorithm monitor."""
        self.monitor.reset()


class EnvData(object):
    """Class to store environment tracking data.

    Attributes
    ----------
    rollout_cnt : Int
        number of rollouts performed on environment.
    """

    def __init__(self):
        """Initialize attributes."""
        self.rollout_cnt = 0


class AlgoData(object):
    """Class used to store algorithm tracking data.

    Attributes
    ----------
    optimize_start : Float
        Start time of the optimization.
    optimize_time : Float
        Start time of intermediate runs.
    step_cnt : Int
        Number of steps performed since initialization.
    rollout_cnts : List
        Number of rollouts during one step.
    parameters : List
        List of parameters found during optimization.
    traces : List
        List of traces for parameters.
    rewards : List
        List of rewards for parameters.
    """

    def __init__(self):
        """Initialize attributes."""
        self.reset()

    def reset(self):
        """Reset monitor data."""
        self.optimize_start = 0
        self.optimize_time = 0

        self.step_cnt = 0
        self.rollout_cnts = []

        self.parameters = []
        self.traces = []
        self.rewards = []
