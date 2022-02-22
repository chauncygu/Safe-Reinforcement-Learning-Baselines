import multiprocessing as mp
from queue import Empty
import time
from rl_safety_algorithms.common.loggers import colorize


def spawn_runner_in_new_process(pid: int,
                                task_queue: mp.Queue,
                                result_queue: mp.Queue = None,
                                verbose: bool = True):
    """A child process that receives seeds from a queue and runs the passed function.

    Parameters
    ----------
    pid:
        The process ID.
    task_queue:
        The input queue that holds the seeds.
    result_queue:
        Define for tests purposes. Store values into the queue and evaluate independent runs.
    verbose: bool
        Print debug information to console if True.

    Returns
    -------
    None

    """
    # create runner instance in child processes' space
    runner = TaskRunner(pid=pid,
                        task_queue=task_queue,
                        result_queue=result_queue,
                        verbose=verbose)
    runner.run()


class Task:
    def __init__(self, id, target_function, kwargs):
        self.id = id
        self.target_function = target_function
        self.kwargs = kwargs


class TaskRunner:
    def __init__(self, pid, task_queue, result_queue, verbose):
        self.pid = pid
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.done = False
        self.verbose = verbose
        print('Started runner with pid = {}'.format(pid)) if verbose else None

    def run(self):
        while not self.done:
            try:
                # wait one sec, if queue is empty raise an error
                received_task = self.task_queue.get(timeout=1)
                assert isinstance(received_task, Task)
                _id = received_task.id
                target_fn = received_task.target_function
                kwargs = received_task.kwargs

                print(colorize(f'INFO: pid {self.pid} executes with seed={_id}',
                               color='green', bold=True))
                # f'\nkwargs: {received_task.kwargs}')
                # call the target function with kwargs
                kwargs.update(seed=_id)
                result = target_fn(**kwargs)

                # puts results into output queue (for tests purposes)
                if self.result_queue:
                    self.result_queue.put(result)

            except Empty:
                self.done = True
                print(f'Stop queue runner with pid = {self.pid}.')


class Scheduler:
    def __init__(self,
                 num_cores: int,
                 verbose: bool = False,
                 ):
        """The scheduler manages the tasks which are executed on a machine.
        The tasks are distributed over all available cores.

        Parameters
        ----------
        num_cores
        verbose
        """
        self.verbose = verbose
        self.num_cores = num_cores
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def fetch_results(self):
        done = False
        outputs = list()
        while not done:
            try:
                # wait one sec, if queue is empty raise an error
                fetched_res = self.result_queue.get(timeout=1)
                outputs.append(fetched_res)
            except Empty:
                done = True
        return outputs

    def fill(self,
             tasks: list
             ) -> None:
        """Fill the queue with tasks."""
        [self.task_queue.put(t) for t in tasks]

    def run(self):
        # Spawn child processes
        child_processes = [
            mp.Process(target=spawn_runner_in_new_process,
                       args=(
                       pid, self.task_queue, self.result_queue, self.verbose))
            for pid in range(self.num_cores)
        ]
        for cp in child_processes:  # start child processes
            cp.start()
            time.sleep(0.1)
        print(colorize('INFO: All workers started.',
                       color='green', highlight=True)) if self.verbose else None

        [cp.join() for cp in child_processes]  # join workers
        print('INFO: All workers joined.') if self.verbose else None
