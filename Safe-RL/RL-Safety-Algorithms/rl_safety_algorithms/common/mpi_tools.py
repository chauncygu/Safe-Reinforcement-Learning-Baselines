""" Message Passing Interface (MPI) tools for parallel computing.

from:
https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_tools.py
and
https://github.com/openai/spinningup/blob/master/spinup/utils/mpi_pytorch.py
"""
from mpi4py import MPI
import subprocess
import sys
import numpy as np
import os
import torch


def setup_torch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    old_num_threads = torch.get_num_threads()
    # decrease number of torch threads for MPI
    if old_num_threads > 1 and num_procs() > 1:
        fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
        torch.set_num_threads(fair_num_threads)
        print(f'Proc {proc_id()}: Decreased number of Torch threads from '
              f'{old_num_threads} to {torch.get_num_threads()}', flush=True)


def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs() > 1:
        for p in module.parameters():
            p_grad_numpy = p.grad.numpy()  # numpy view of tensor data
            avg_p_grad = mpi_avg(p.grad)
            p_grad_numpy[:] = avg_p_grad[:]


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs() > 1:
        for p in module.parameters():
            p_numpy = p.data.numpy()
            broadcast(p_numpy)


def mpi_fork(
        n: int,
        bind_to_core=False,
        use_number_of_threads=False
) -> bool:
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.

        use_number_of_threads (bool): If you want Open MPI to default to the
        number of hardware threads instead of the number of processor cores, use
        the --use-hwthread-cpus option.

    Returns:
        bool
            True if process is parent process of MPI

    Usage:
        if mpi_fork(n):  # forks the current script and calls MPI
            sys.exit()   # exit single thread python process

    """
    is_parent = False
    # Check if MPI is already setup..
    if n > 1 and os.getenv("IN_MPI") is None:
        # MPI is not yet set up: quit parent process and start N child processes
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        if use_number_of_threads:
            args += ["--use-hwthread-cpus"]
        args += [sys.executable] + sys.argv
        # This is the parent process, spawn sub-processes..
        subprocess.check_call(args, env=env)
        is_parent = True
    return is_parent


def msg(m, string=''):
    print(
        ('Message from %d: %s \t ' % (MPI.COMM_WORLD.Get_rank(), string)) + str(
            m))


def is_root_process() -> bool:
    return True if MPI.COMM_WORLD.Get_rank() == 0 else False


def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def gather(*args, **kwargs):
    return MPI.COMM_WORLD.Gather(*args, **kwargs)


def gather_and_stack(x: np.ndarray) -> np.array:
    """Gather values from all tasks and return flattened list.
    Note: only the root process owns valid data.

    Parameters
        x
            1-D array of size N

    Returns
        list of size N * MPI_world_size
    """
    if num_procs() == 1:
        return x
    assert x.ndim == 1, 'Only lists or 1D-arrays supported.'
    buf = None
    size = num_procs()
    length = x.shape[0]
    # if proc_id() == 0:
    buf = np.empty([size, length], dtype=np.float32)
    gather(x, buf, root=0)
    return buf.flatten()


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    assert isinstance(x, np.ndarray)
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_avg(x):
    """Average a scalar or numpy vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def mpi_max(x):
    """Determine global maximum of scalar or numpy array over MPI processes."""
    return mpi_op(x, MPI.MAX)


def mpi_min(x):
    """Determine global minimum of scalar or numpy array over MPI processes."""
    return mpi_op(x, MPI.MIN)


def mpi_op(x, op):
    if num_procs() == 1:
        return x
    # try:
    #     assert isinstance(x, np.ndarray) or isinstance(x, list)
    # except AssertionError:
    #     print(f'x: {x}')
    #     print(f'type: {type(x)}')
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg_torch_tensor(x) -> None:
    """Average a torch tensor over MPI processes.

    Since torch and numpy share same memory space, tensors of dim > 0 can be
    be manipulated through call by reference, scalars must be assigned.
    """
    assert isinstance(x, torch.Tensor)
    if num_procs() > 1:
        # tensors can be manipulated in-place
        if len(x.shape) > 0:
            x_numpy = x.numpy()  # numpy view of tensor data
            avg_x_numpy = mpi_avg(x_numpy)
            x_numpy[:] = avg_x_numpy[:]  # in-place memory update
        else:
            # scalars (tensors of dim = 0) must be assigned
            raise NotImplementedError


def mpi_statistics_scalar(x, with_min_and_max=False) -> tuple:
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean) ** 2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    else:
        return mean, std
