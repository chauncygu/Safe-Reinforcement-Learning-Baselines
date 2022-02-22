from abc import ABC, abstractmethod
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from . import defaults
from .config import BaseConfig, Configurable
from .normalization import Normalizer
from .train import epochal_training
from .torch_util import device, Module, mlp


class BaseModel(ABC):
    @abstractmethod
    def sample(self, states, actions):
        """
        Returns a sample of (s', r) given (s, a)
        """
        pass


class BatchedLinear(nn.Module):
    """For efficient MLP ensembles with batched matrix multiplies"""
    def __init__(self, ensemble_size, in_features, out_features, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(ensemble_size, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        has_bias = self.bias is not None
        l = nn.Linear(self.in_features, self.out_features, bias=has_bias)
        for i in range(self.ensemble_size):
            l.reset_parameters()
            self.weight.data[i].copy_(l.weight.data)
            if has_bias:
                self.bias.data[i].copy_(l.bias.data)

    def forward(self, input):
        assert len(input.shape) == 3
        assert input.shape[0] == self.ensemble_size
        return torch.bmm(input, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)


class BatchedGaussianEnsemble(Configurable, Module, BaseModel):
    class Config(BaseConfig):
        ensemble_size = 5
        hidden_dim = 200
        trunk_layers = 2
        head_hidden_layers = 1
        activation = 'relu'
        init_min_log_var = -10.0
        init_max_log_var = 1.0
        log_var_bound_weight = 0.01
        batch_size = 256
        learning_rate = 1e-3

    def __init__(self, config, state_dim, action_dim,
                 device=device, optimizer_factory=defaults.OPTIMIZER):
        Configurable.__init__(self, config)
        Module.__init__(self)

        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        output_dim = state_dim + 1

        self.min_log_var = nn.Parameter(torch.full([output_dim], self.init_min_log_var, device=device))
        self.max_log_var = nn.Parameter(torch.full([output_dim], self.init_max_log_var, device=device))
        self.state_normalizer = Normalizer(state_dim)

        layer_factory = lambda n_in, n_out: BatchedLinear(self.ensemble_size, n_in, n_out)
        trunk_dims = [input_dim] + [self.hidden_dim] * self.trunk_layers
        head_dims = [self.hidden_dim] * (self.head_hidden_layers + 1) + [output_dim]
        self.trunk = mlp(trunk_dims, layer_factory=layer_factory, activation=self.activation,
                         output_activation=self.activation)
        self.diff_head = mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
        self.log_var_head = mlp(head_dims, layer_factory=layer_factory, activation=self.activation)
        self.to(device)
        self.optimizer = optimizer_factory([
            *self.trunk.parameters(),
            *self.diff_head.parameters(),
            *self.log_var_head.parameters(),
            self.min_log_var, self.max_log_var
        ], lr=self.learning_rate)


    @property
    def total_batch_size(self):
        return self.ensemble_size * self.batch_size

    def _forward1(self, states, actions, index):
        normalized_states = self.state_normalizer(states)
        inputs = torch.cat([normalized_states, actions], dim=-1)
        batch_size = inputs.shape[0]
        shared_hidden = unbatched_forward(self.trunk, inputs, index)
        diffs = unbatched_forward(self.diff_head, shared_hidden, index)
        means = diffs + torch.cat([states, torch.zeros([batch_size, 1], device=device)], dim=1)
        log_vars = unbatched_forward(self.log_var_head, shared_hidden, index)
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    def _forward_all(self, states, actions):
        normalized_states = self.state_normalizer(states)
        inputs = torch.cat([normalized_states, actions], dim=-1)
        batch_size = inputs.shape[1]
        shared_hidden = self.trunk(inputs)
        diffs = self.diff_head(shared_hidden)
        means = diffs + torch.cat([states, torch.zeros([self.ensemble_size, batch_size, 1], device=device)], dim=-1)
        log_vars = self.log_var_head(shared_hidden)
        log_vars = self.max_log_var - F.softplus(self.max_log_var - log_vars)
        log_vars = self.min_log_var + F.softplus(log_vars - self.min_log_var)
        return means, log_vars

    def _rebatch(self, x):
        total_batch_size = len(x)
        assert total_batch_size % self.ensemble_size == 0, f'{total_batch_size} not divisible by {self.ensemble_size}'
        batch_size = total_batch_size // self.ensemble_size
        remaining_dims = tuple(x.shape[1:])
        return x.reshape(self.ensemble_size, batch_size, *remaining_dims)

    def compute_loss(self, states, actions, targets):
        inputs = [states, actions, targets]
        total_batch_size = len(targets)
        remainder = total_batch_size % self.ensemble_size
        if remainder != 0:
            nearest = total_batch_size - remainder
            inputs = [x[:nearest] for x in inputs]

        states, actions, targets = [self._rebatch(x) for x in inputs]
        means, log_vars = self._forward_all(states, actions)
        inv_vars = torch.exp(-log_vars)
        squared_errors = torch.sum((targets - means)**2 * inv_vars, dim=-1)
        log_dets = torch.sum(log_vars, dim=-1)
        mle_loss = torch.mean(squared_errors + log_dets)
        return mle_loss + self.log_var_bound_weight * (self.max_log_var.sum() - self.min_log_var.sum())

    def fit(self, buffer, steps=None, epochs=None, progress_bar=False, **kwargs):
        n = len(buffer)
        states, actions, next_states, rewards = buffer.get()[:4]
        self.state_normalizer.fit(states)
        targets = torch.cat([next_states, rewards.unsqueeze(1)], dim=1)

        if steps is not None:
            assert epochs is None, 'Cannot pass both steps and epochs'
            losses = []
            for _ in (trange if progress_bar else range)(steps):
                # indices = random_indices(n, size=self.total_batch_size, replace=False)
                indices = torch.randint(n, [self.total_batch_size], device=device)
                loss = self.compute_loss(states[indices], actions[indices], targets[indices])
                losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return losses
        elif epochs is not None:
            # Because of the batching, each model only sees about 1/ensemble_size fraction of the data per epoch.
            # Therefore we will multiply the number of epochs by ensemble_size.
            adjusted_epochs = self.ensemble_size * epochs
            return epochal_training(self.compute_loss, self.optimizer, [states, actions, targets],
                                    epochs=adjusted_epochs,
                                    batch_size=self.total_batch_size, **kwargs)
        else:
            raise ValueError('Must pass steps or epochs')

    def sample(self, states, actions):
        index = random.randrange(self.ensemble_size)
        means, log_vars = self._forward1(states, actions, index)
        stds = torch.exp(log_vars).sqrt()
        samples = means + stds * torch.randn_like(means)
        return samples[:,:-1], samples[:,-1]

    # Get all models' means on the same set of states and actions
    def means(self, states, actions):
        states = states.repeat(self.ensemble_size, 1, 1)
        actions = actions.repeat(self.ensemble_size, 1, 1)
        means, _ = self._forward_all(states, actions)
        return means[:,:,:-1], means[:,:,-1]

    # Get average of models' means
    def mean(self, states, actions):
        next_state_means, reward_means = self.means(states, actions)
        return next_state_means.mean(dim=0), reward_means.mean(dim=0)



# Special forward for nn.Sequential modules which contain BatchedLinear layers,
# for when we only want to use one of the models.
def unbatched_forward(batched_sequential, input, index):
    for layer in batched_sequential:
        if isinstance(layer, BatchedLinear):
            input = F.linear(input, layer.weight[index], layer.bias[index])
        else:
            input = layer(input)
    return input