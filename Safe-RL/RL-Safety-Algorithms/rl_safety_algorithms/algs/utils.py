import numpy as np
import gym
from torch.optim import Adam, SGD
import time
import torch
from rl_safety_algorithms.algs import core
from rl_safety_algorithms.common import loggers
import os


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, eps=1e-6):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)

    nsteps: (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.
            Also probably don't play with this hyperparameter.
    """
    x = torch.zeros_like(b)
    r = b - Avp(x)
    p = r.clone()
    rdotr = torch.dot(r, r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    verbose = False

    for i in range(nsteps):
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = Avp(p)
        alpha = rdotr / (torch.dot(p, z) + eps)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        mu = new_rdotr / (rdotr + eps)
        p = r + mu * p
        rdotr = new_rdotr

    return x


def get_flat_gradients_from(model):
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad
            grads.append(g.view(-1))  # flatten tensor and append
    assert grads is not [], 'No gradients were found in model parameters.'

    return torch.cat(grads)


def get_flat_params_from(model):
    flat_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            d = param.data
            d = d.view(-1)  # flatten tensor
            flat_params.append(d)
    assert flat_params is not [], 'No gradients were found in model parameters.'

    return torch.cat(flat_params)


def set_flat_grads_to_model(model, grads: torch.Tensor):
    # assert isinstance(grads, torch.Tensor)
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = grads[i:i + size]

            # set new gradients
            try:
                param.grad.data = new_values.view(orig_size)
            except AttributeError:
                # AttributeError: 'NoneType' object has no attribute 'data'
                # in case grad is None
                param.grad = new_values.view(orig_size)
            i += size  # increment array position
    assert i == len(grads), f'Lengths do not match: {i} vs. {len(grads)}'


def set_param_values_to_model(model, vals):
    assert isinstance(vals, torch.Tensor)
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i:i + size]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += size  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'

# def Mvp(vec, ac, _obs):
#     """ Matrix-vector product (used for efficient Fisher-vector product calc)
#         from stable-baselines:
#         https://github.com/hill-a/stable-baselines/blob/c4c31cb5687800cf102054f0935aa09e7545159f/stable_baselines/trpo_mpi/trpo_mpi.py#L356
#
#         For details on sub-sampling: see John Schulman thesis (pp. 40)
#         http://joschu.net/docs/thesis.pdf
#     """
#     kl = get_kl(_obs[::5])  # sub-sampling
#     mean_kl = kl.mean()
#
#     grads = torch.autograd.grad(mean_kl, ac.pi.net.parameters(),
#                                 create_graph=True, retain_graph=True)
#     flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
#
#     kl_p = (flat_grad_kl * torch.nn.Parameter(vec.data)).sum()
#     grads = torch.autograd.grad(kl_p, ac.pi.net.parameters())
#     # contiguous indicating, if the memory is contiguously stored or not
#     flat_grad_grad_kl = torch.cat(
#         [grad.contiguous().view(-1) for grad in grads]).data
#
#     # todo: perform damping in calling function
#     return flat_grad_grad_kl  # + p * cg_damping
