from abc import ABC, abstractmethod

import gym
from gym.spaces import Box, Discrete, Space
import torch
from torch import distributions as td

from .torch_util import torchify, Module, device
from .squashed_gaussian import SquashedGaussian


class BasePolicy(ABC):
    @abstractmethod
    def act(self, states, eval): pass

    def act1(self, state, eval=False):
        return self.act(torch.unsqueeze(state, 0), eval)[0]


class UniformPolicy(BasePolicy):
    def __init__(self, env_or_action_space):
        if isinstance(env_or_action_space, gym.Env):
            action_space = env_or_action_space.action_space
        elif isinstance(env_or_action_space, gym.Space):
            action_space = env_or_action_space
        else:
            raise ValueError('Must pass env or action space')

        if isinstance(action_space, Box):
            self.low = torchify(action_space.low, to_device=True)
            self.high = torchify(action_space.high, to_device=True)
            self.shape = list(action_space.shape)
            self.discrete = False
        elif isinstance(action_space, Discrete):
            self.n = action_space.n
            self.discrete = True
        else:
            raise NotImplementedError(f'Unsupported action space: {action_space}')

    def act(self, states, eval):
        batch_size = len(states)
        if self.discrete:
            return torch.randint(self.n, size=(batch_size,), device=device)
        else:
            return self.low + torch.rand(batch_size, *self.shape, device=device) * (self.high - self.low)

    def prob(self, actions):
        batch_size = len(actions)
        if self.discrete:
            assert actions.dim() == 1
            p = 1./self.n
        else:
            assert actions.dim() == 2
            p = 1./torch.prod(self.high - self.low)
        return torch.full([batch_size], p, device=device)

    def log_prob(self, actions):
        return torch.log(self.prob(actions))


class TorchPolicy(BasePolicy, Module):
    def __init__(self, net):
        Module.__init__(self)
        self.net = net
        self.use_special_eval = False

    @abstractmethod
    def _distr(self, *network_outputs): pass

    def distr(self, states):
        return self._distr(self.net(states))

    @abstractmethod
    def _special_eval(self, distr):
        raise NotImplementedError

    def act(self, states, eval):
        with torch.no_grad():
            distr = self.distr(states)
        return self._special_eval(distr) if self.use_special_eval else distr.sample()


class SquashedGaussianPolicy(TorchPolicy):
    def __init__(self, net, log_std_bounds=(-20,2), std_multiplier=1.0):
        super().__init__(net)
        self.log_std_bounds = log_std_bounds
        self.std_multiplier = std_multiplier

    def _distr(self, net_out):
        mu, log_std = net_out.chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + (log_std_max - log_std_min) * torch.sigmoid(log_std)
        # log_std = log_std.clamp(log_std_min, log_std_max)
        std = log_std.exp() * self.std_multiplier
        return td.Independent(SquashedGaussian(mu, std, validate_args=True), 1)

    def _special_eval(self, distr):
        return distr.mean