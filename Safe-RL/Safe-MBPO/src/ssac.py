import copy
import math
import random

import torch
from torch import nn

from .config import BaseConfig, Configurable, Optional
from .defaults import ACTOR_LR, OPTIMIZER
from .log import default_log as log
from .policy import BasePolicy, SquashedGaussianPolicy
from .torch_util import device, Module, mlp, update_ema, freeze_module
from .util import pythonic_mean


class CriticEnsemble(Configurable, Module):
    class Config(BaseConfig):
        n_critics = 2
        hidden_layers = 2
        hidden_dim = 256
        learning_rate = 3e-4

    def __init__(self, config, state_dim, action_dim):
        Configurable.__init__(self, config)
        Module.__init__(self)
        dims = [state_dim + action_dim, *([self.hidden_dim] * self.hidden_layers), 1]
        self.qs = torch.nn.ModuleList([
            mlp(dims, squeeze_output=True) for _ in range(self.n_critics)
        ])
        self.optimizer = torch.optim.Adam(self.qs.parameters(), lr=self.learning_rate)

    def all(self, state, action):
        sa = torch.cat([state, action], 1)
        return [q(sa) for q in self.qs]

    def min(self, state, action):
        return torch.min(*self.all(state, action))

    def mean(self, state, action):
        return pythonic_mean(self.all(state, action))

    def random_choice(self, state, action):
        sa = torch.cat([state, action], 1)
        return random.choice(self.qs)(sa)


class SSAC(BasePolicy, Module):
    class Config(BaseConfig):
        discount = 0.99
        init_alpha = 1.0
        autotune_alpha = True
        target_entropy = Optional(float)
        use_log_alpha_loss = True
        deterministic_backup = False
        critic_update_multiplier = 1
        actor_lr = ACTOR_LR
        critic_cfg = CriticEnsemble.Config()
        tau = 0.005
        batch_size = 256
        hidden_dim = 256
        hidden_layers = 2
        update_violation_cost = True

    def __init__(self, config, state_dim, action_dim, horizon,
                 optimizer_factory=OPTIMIZER):
        Configurable.__init__(self, config)
        Module.__init__(self)
        self.horizon = horizon
        self.violation_cost = 0.0

        self.actor = SquashedGaussianPolicy(mlp(
            [state_dim, *([self.hidden_dim] * self.hidden_layers), action_dim*2]
        ))
        self.critic = CriticEnsemble(self.critic_cfg, state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        freeze_module(self.critic_target)

        self.actor_optimizer = optimizer_factory(self.actor.parameters(), lr=self.actor_lr)

        log_alpha = torch.tensor(math.log(self.init_alpha), device=device, requires_grad=True)
        self.log_alpha = log_alpha
        if self.autotune_alpha:
            self.alpha_optimizer = optimizer_factory([self.log_alpha], lr=self.actor_lr)
        if self.target_entropy is None:
            self.target_entropy = -action_dim   # set target entropy to -dim(A)

        self.criterion = nn.MSELoss()

        self.register_buffer('total_updates', torch.zeros([]))

    def act(self, states, eval):
        return self.actor.act(states, eval)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def violation_value(self):
        return -self.violation_cost / (1. - self.discount)

    def update_r_bounds(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        if self.update_violation_cost:
            self.violation_cost = (r_max - r_min) / self.discount**self.horizon - r_max
        log.message(f'r bounds: [{r_min, r_max}], C = {self.violation_cost}')

    def critic_loss(self, obs, action, next_obs, reward, done):
        reward = reward.clamp(self.r_min, self.r_max)
        target = super().compute_target(next_obs, reward, done)
        if done.any():
            target[done] = self.terminal_value
        return self.critic_loss_given_target(obs, action, target)

    def compute_target(self, next_obs, reward, done, violation):
        with torch.no_grad():
            distr = self.actor.distr(next_obs)
            next_action = distr.sample()
            log_prob = distr.log_prob(next_action)
            next_value = self.critic_target.min(next_obs, next_action)
            if not self.deterministic_backup:
                next_value = next_value - self.alpha.detach() * log_prob
            q = reward + self.discount * (1. - done.float()) * next_value
            q[violation] = self.violation_value
            return q

    def critic_loss_given_target(self, obs, action, target):
        qs = self.critic.all(obs, action)
        return pythonic_mean([self.criterion(q, target) for q in qs])

    def critic_loss(self, obs, action, next_obs, reward, done, violation):
        target = self.compute_target(next_obs, reward, done, violation)
        return self.critic_loss_given_target(obs, action, target)

    def update_critic(self, *critic_loss_args):
        critic_loss = self.critic_loss(*critic_loss_args)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        update_ema(self.critic_target, self.critic, self.tau)
        return critic_loss.detach()

    def actor_loss(self, obs, include_alpha=True):
        distr = self.actor.distr(obs)
        action = distr.rsample()
        log_prob = distr.log_prob(action)
        actor_Q = self.critic.random_choice(obs, action)
        alpha = self.alpha
        actor_loss = torch.mean(alpha.detach() * log_prob - actor_Q)
        if include_alpha:
            multiplier = self.log_alpha if self.use_log_alpha_loss else alpha
            alpha_loss = -multiplier * torch.mean(log_prob.detach() + self.target_entropy)
            return [actor_loss, alpha_loss]
        else:
            return [actor_loss]

    def update_actor_and_alpha(self, obs):
        losses = self.actor_loss(obs, include_alpha=self.autotune_alpha)
        optimizers = [self.actor_optimizer, self.alpha_optimizer] if self.autotune_alpha else \
                     [self.actor_optimizer]
        assert len(losses) == len(optimizers)
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def update(self, replay_buffer):
        assert self.critic_update_multiplier >= 1
        for _ in range(self.critic_update_multiplier):
            samples = replay_buffer.sample(self.batch_size)
            self.update_critic(*samples)
        self.update_actor_and_alpha(samples[0])
        self.total_updates += 1