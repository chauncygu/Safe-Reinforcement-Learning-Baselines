from math import ceil
import random
import string

import numpy as np
import torch

from .torch_util import device


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def pythonic_mean(x):
    return sum(x) / len(x)


def random_string(n, include_lowercase=True, include_uppercase=False, include_digits=False):
    alphabet = ''
    if include_lowercase:
        alphabet += string.ascii_lowercase
    if include_uppercase:
        alphabet += string.ascii_uppercase
    if include_digits:
        alphabet += string.digits
    return ''.join(random.choices(alphabet, k=n))


def unique_value(name):
    """Creates a unique value with the given name. The value has its own class, of which it is the only instance."""
    class _UniqueClass:
        def __str__(self):
            return name
        def __repr__(self):
            return name
    _UniqueClass.__name__ = name
    return _UniqueClass()


def discounted_sum(rewards, discount):
    return torch.sum(rewards * discount**torch.arange(len(rewards), dtype=torch.float, device=device))


def batch_iterator(args, batch_size=256, shuffle=False):
    if type(args) in {list, tuple}:
        multi_arg = True
        n = len(args[0])
        for i, arg_i in enumerate(args):
            assert isinstance(arg_i, torch.Tensor)
            assert len(arg_i) == n
    else:
        multi_arg = False
        n = len(args)

    indices = torch.randperm(n) if shuffle else torch.arange(n)

    n_batches = ceil(float(n) / batch_size)
    for batch_index in range(n_batches):
        batch_start = batch_size * batch_index
        batch_end = min(batch_size * (batch_index + 1), n)
        batch_indices = indices[batch_start:batch_end]
        if multi_arg:
            yield tuple(arg[batch_indices] for arg in args)
        else:
            yield args[batch_indices]


def batch_map(fn, args, batch_size=1000, cat_dim=0):
    if type(args) in {list, tuple}:
        results = [fn(*batch) for batch in batch_iterator(args, batch_size=batch_size)]
    else:
        results = [fn(batch) for batch in batch_iterator(args, batch_size=batch_size)]
    return torch.cat(results, dim=cat_dim)