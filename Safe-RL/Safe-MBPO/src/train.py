from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from tqdm import trange

from .log import default_log as log
from .torch_util import torchify


def supervised_loss(forward, criterion):
    return lambda x, y: criterion(forward(x), y)


# For some reason this isn't already implemented in PyTorch
class L2Loss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__(reduction=reduction)

    def forward(self, input, target):
        assert input.shape == target.shape, f'Shape mismatch: {input.shape} and {target.shape}'
        l2_dists = torch.dist(input, target)
        if self.reduction == 'mean':
            return l2_dists.mean()
        elif self.reduction == 'sum':
            return l2_dists.sum()
        elif self.reduction == 'none':
            return l2_dists
        else:
            raise NotImplementedError(f'Unknown reduction {self.reduction}')


def default_optimizer_factory(params):
    return torch.optim.AdamW(params, weight_decay=1e-4)

def get_optimizer(module_or_params, optimizer=None):
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    else:
        if optimizer is None:
            optimizer_factory = default_optimizer_factory
        else:
            assert callable(optimizer)
            optimizer_factory = optimizer
        parameters = module_or_params.parameters() if isinstance(module_or_params, nn.Module) else module_or_params
        return optimizer_factory(parameters)


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def epochal_training(compute_loss, optimizer, data, epochs, batch_size=256, max_grad_norm=None,
                     post_epoch_callback=None, post_step_callback=None,
                     progress_bar=False, verbose=False):
    def one_step(batch):
        loss = compute_loss(*batch)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            for param_group in optimizer.param_groups:
                nn.utils.clip_grad_norm_(param_group['params'], max_grad_norm)
        optimizer.step()
        return loss.item()

    if torch.is_tensor(data):
        data = (data,)
    n = len(data[0])
    for i, data_i in enumerate(data):
        assert len(data_i) == n

    data = [torchify(data_i) for data_i in data]

    n_batches = ceil(float(n) / batch_size)
    iter_range_fn = trange if progress_bar else range
    losses = []
    for epoch_index in range(epochs):
        indices = torch.randperm(n)
        epoch_losses = []
        for batch_index in iter_range_fn(n_batches):
            batch_start = batch_size * batch_index
            batch_end = min(batch_size * (batch_index + 1), n)
            batch_indices = indices[batch_start:batch_end]
            loss_val = one_step([component[batch_indices] for component in data])
            epoch_losses.append(loss_val)
            if post_step_callback is not None:
                post_step_callback(epoch_index, batch_index, n_batches)
        avg_epoch_loss = float(np.mean(epoch_losses))
        losses.append(avg_epoch_loss)

        if verbose:
            log.message(f'Finished epoch {epoch_index+1}/{epochs}. Average loss: {avg_epoch_loss}')
        if post_epoch_callback is not None:
            post_epoch_callback(epoch_index + 1)

    return losses