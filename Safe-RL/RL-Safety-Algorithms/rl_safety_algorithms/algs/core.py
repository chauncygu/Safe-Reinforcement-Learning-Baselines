""" Core ingredients for RL algorithms.

Author:     Sven Gronauer (sven.gronauer@tum.de)
based on:   Spinning Up's Vanilla Policy Gradient
            https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py
"""
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import abc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from rl_safety_algorithms.common.online_mean_std import OnlineMeanStd
from rl_safety_algorithms.algs.vtrace import calculate_v_trace
import rl_safety_algorithms.common.mpi_tools as mpi_tools

registered_actors = dict()  # global dict that holds pointers to functions 


def get_optimizer(opt: str, module: torch.nn.Module, lr: float):
    """ Returns an initialized optimizer from PyTorch."""
    assert hasattr(optim, opt), f'Optimizer={opt} not found in torch.'
    optimizer = getattr(optim, opt)

    return optimizer(module.parameters(), lr=lr)


def initialize_layer(
        init_function: str,
        layer: torch.nn.Module
):
    if init_function == 'kaiming_uniform':  # this the default!
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    # glorot is also known as xavier uniform
    elif init_function == 'glorot' or init_function == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':  # matches values from baselines repo.
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise NotImplementedError
    # print(layer)
    # print(layer.weight)


def register_actor(actor_name):
    """ register actor into global dict"""
    def wrapper(func):
        registered_actors[actor_name] = func
        return func
    return wrapper


def get_registered_actor_fn(actor_type: str, distribution_type: str):
    assert distribution_type == 'categorical' or distribution_type == 'gaussian'
    actor_fn = actor_type + '_' + distribution_type
    msg = f'Did not find: {actor_fn} in registered actors.'
    assert actor_fn in registered_actors, msg
    return registered_actors[actor_fn]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def convert_str_to_torch_functional(activation):
    if isinstance(activation, str):  # convert string to torch functional
        activations = {
            'identity': nn.Identity,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'softplus': nn.Softplus,
            'tanh': nn.Tanh
        }
        assert activation in activations
        activation = activations[activation]
    assert issubclass(activation, torch.nn.Module)
    return activation


def build_mlp_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform'
):
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization, affine_layer)
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[
           ::-1]


# ====================================
#       Algorithm Classes
# ====================================


class Algorithm(abc.ABC):
    @abc.abstractmethod
    def learn(self) -> tuple:
        pass

    @abc.abstractmethod
    def log(self, epoch: int):
        pass

    @abc.abstractmethod
    def update(self):
        pass


class PolicyGradientAlgorithm(Algorithm, abc.ABC):

    @abc.abstractmethod
    def roll_out(self):
        """collect data and store to experience buffer."""
        pass


class ConstrainedPolicyGradientAlgorithm(abc.ABC):
    """ Abstract base class for Lagrangian-TRPO and Lagrangian-PPO."""
    def __init__(self,
                 cost_limit: float,
                 use_lagrangian_penalty: bool,
                 lagrangian_multiplier_init: float,
                 lambda_lr: float,
                 lambda_optimizer: str
                 ):
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.use_lagrangian_penalty = use_lagrangian_penalty

        init_value = max(lagrangian_multiplier_init, 1e-5)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value),
            requires_grad=True)
        self.lambda_range_projection = torch.nn.ReLU()

        # fetch optimizer from PyTorch optimizer package
        assert hasattr(optim, lambda_optimizer), \
            f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt([self.lagrangian_multiplier, ],
                                          lr=lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost):
        """Penalty loss for Lagrange multiplier."""
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)

    def update_lagrange_multiplier(self, ep_costs):
        """ Update Lagrange multiplier (lambda)
            Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
            are already averaged across MPI processes.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(ep_costs)
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]


# ====================================
#       Actor Modules
# ====================================


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, weight_initialization, shared=None):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shared = shared
        self.weight_initialization = weight_initialization

    def dist(self, obs) -> torch.distributions.Distribution:
        raise NotImplementedError

    def log_prob_from_dist(self, pi, act) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs, act=None) -> tuple:
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.dist(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_dist(pi, act)
        return pi, logp_a

    def sample(self, obs) -> tuple:
        raise NotImplementedError

    def predict(self, obs) -> tuple:
        """ Predict action based on observation without exploration noise.
            Use this method for evaluation purposes. """
        return self.sample(obs)


@register_actor("mlp_categorical")
class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,
                 weight_initialization, shared=None):
        super().__init__(obs_dim, act_dim, weight_initialization, shared=shared)
        if shared is not None:
            raise NotImplementedError
        self.net = build_mlp_network(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            weight_initialization=weight_initialization
        )

    def dist(self, obs) -> torch.distributions.Distribution:
        logits = self.net(obs)
        return Categorical(logits=logits)

    def log_prob_from_dist(self, pi, act) -> torch.Tensor:
        return pi.log_prob(act)

    def sample(self, obs) -> tuple:
        # frac is necessary for epsilon greedy
        # eps_threshold = np.max([self.current_eps, self.min_eps])

        dist = self.dist(obs)
        a = dist.sample()
        logp_a = self.log_prob_from_dist(dist, a)

        return a, logp_a


@register_actor("mlp_gaussian")
class MLPGaussianActor(Actor):
    def __init__(
            self,
            obs_dim,
            act_dim,
            hidden_sizes,
            activation,
            weight_initialization,
            shared=None):
        super().__init__(obs_dim, act_dim, weight_initialization)
        log_std = np.log(0.5) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std),
                                          requires_grad=False)

        if shared is not None:  # use shared layers
            action_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.net = nn.Sequential(shared, action_head, nn.Identity())
        else:
            layers = [self.obs_dim] + list(hidden_sizes) + [self.act_dim]
            self.net = build_mlp_network(
                layers,
                activation=activation,
                weight_initialization=weight_initialization
            )

    def dist(self, obs):
        mu = self.net(obs)
        return Normal(mu, self.std)

    def log_prob_from_dist(self, pi, act) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def sample(self, obs):
        pi = self.dist(obs)
        a = pi.sample()
        logp_a = self.log_prob_from_dist(pi, a)

        return a, logp_a

    def set_log_std(self, frac):
        """ To support annealing exploration noise.
            frac is annealing from 1. to 0 over course of training"""
        assert 0 <= frac <= 1
        new_stddev = 0.499 * frac + 0.01  # annealing from 0.5 to 0.01
        # new_stddev = 0.3 * frac + 0.2  # linearly anneal stddev from 0.5 to 0.2
        log_std = np.log(new_stddev) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std),
                                          requires_grad=False)

    @property
    def std(self):
        """ Standard deviation of distribution."""
        return torch.exp(self.log_std)

    def predict(self, obs):
        """ Predict action based on observation without exploration noise.
            Use this method for evaluation purposes. """
        action = self.net(obs)
        log_p = torch.ones_like(action)  # avoid type conflicts at evaluation

        return action, log_p


# ====================================
#       Critic Modules
# ====================================


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, shared=None):
        super().__init__()
        if shared is None:
            self.net = build_mlp_network([obs_dim] + list(hidden_sizes) + [1],
                                           activation=activation)
        else:  # use shared layers
            value_head = nn.Linear(hidden_sizes[-1], 1)
            self.net = nn.Sequential(shared, value_head, nn.Identity())

    def forward(self, obs):
        return torch.squeeze(self.net(obs),
                             -1)  # Critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(self,
                 actor_type,
                 observation_space,
                 action_space,
                 use_standardized_obs,
                 use_scaled_rewards,
                 use_shared_weights,
                 ac_kwargs,
                 weight_initialization='kaiming_uniform'
                 ):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) \
            if use_standardized_obs else None
        self.ac_kwargs = ac_kwargs

        # policy builder depends on action space
        if isinstance(action_space, Box):
            distribution_type = 'gaussian'
            act_dim = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            distribution_type = 'categorical'
            act_dim = action_space.n
        else:
            raise ValueError

        obs_dim = observation_space.shape[0]
        layer_units = [obs_dim] + list(ac_kwargs['pi']['hidden_sizes'])
        act = ac_kwargs['pi']['activation']
        if use_shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=act,
                weight_initialization=weight_initialization,
                output_activation=act
            )
        else:
            shared = None

        actor_fn = get_registered_actor_fn(actor_type, distribution_type)
        self.pi = actor_fn(obs_dim=obs_dim,
                           act_dim=act_dim,
                           shared=shared,
                           weight_initialization=weight_initialization,
                           **ac_kwargs['pi'])
        self.v = MLPCritic(obs_dim,
                           shared=shared,
                           **ac_kwargs['val'])

        self.ret_oms = OnlineMeanStd(shape=(1,)) if use_scaled_rewards else None

    def forward(self,
                obs: torch.Tensor
                ) -> tuple:
        return self.step(obs)

    def step(self,
             obs: torch.Tensor
             ) -> tuple:
        """ Produce action, value, log_prob(action).
            If training, this includes exploration noise!

            Expects that obs is not pre-processed.

            Note:
                Training mode can be activated with ac.train()
                Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            v = self.v(obs)
            if self.training:
                a, logp_a = self.pi.sample(obs)
            else:
                a, logp_a = self.pi.predict(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self,
            obs: torch.Tensor
            ) -> np.ndarray:
        return self.step(obs)[0]

    def update(self, frac):
        """update internals of actors

            1) Updates exploration parameters
            + for Gaussian actors update log_std

        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1

        """
        if hasattr(self.pi, 'set_log_std'):
            self.pi.set_log_std(1 - frac)


class ActorCriticWithCosts(ActorCritic):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.c = MLPCritic(
            obs_dim=self.obs_shape[0],
            shared=None,
            **self.ac_kwargs['val'])

    def step(self,
             obs: torch.Tensor
             ) -> tuple:
        """ Produce action, value, log_prob(action).
            If training, this includes exploration noise!

            Note:
                Training mode can be activated with ac.train()
                Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: do the updates at the end of batch!
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            v = self.v(obs)
            c = self.c(obs)

            if self.training:
                a, logp_a = self.pi.sample(obs)
            else:
                a, logp_a = self.pi.predict(obs)

        return a.numpy(), v.numpy(), c.numpy(), logp_a.numpy()


class Buffer:
    def __init__(self,
                 actor_critic: torch.nn.Module,
                 obs_dim: tuple,
                 act_dim: tuple,
                 size: int,
                 gamma: float,
                 lam: float,
                 adv_estimation_method: str,
                 use_scaled_rewards: bool,
                 standardize_env_obs: bool,
                 standardize_advantages: bool,
                 lam_c: float = 0.95,
                 use_reward_penalty: bool = False
                 ):
        """
        A buffer for storing trajectories experienced by an agent interacting
        with the environment, and using Generalized Advantage Estimation (GAE)
        for calculating the advantages of state-action pairs.

        Important Note: Buffer collects only raw data received from environment.
        """
        self.actor_critic = actor_critic
        self.size = size
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.discounted_ret_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.target_val_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.lam_c = lam_c
        self.adv_estimation_method = adv_estimation_method
        self.use_scaled_rewards = use_scaled_rewards
        self.standardize_env_obs = standardize_env_obs
        self.standardize_advantages = standardize_advantages
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

        # variables for cost-based RL
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.cost_val_buf = np.zeros(size, dtype=np.float32)
        self.cost_adv_buf = np.zeros(size, dtype=np.float32)
        self.target_cost_val_buf = np.zeros(size, dtype=np.float32)
        self.use_reward_penalty = use_reward_penalty

        assert adv_estimation_method in ['gae', 'vtrace', 'plain']

    def calculate_adv_and_value_targets(self, vals, rews, lam=None):
        """ Compute the estimated advantage"""

        if self.adv_estimation_method == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            lam = self.lam if lam is None else lam
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            adv = discount_cumsum(deltas, self.gamma * lam)
            value_net_targets = adv + vals[:-1]

        elif self.adv_estimation_method == 'vtrace':
            #  v_s = V(x_s) + \sum^{T-1}_{t=s} \gamma^{t-s}
            #                * \prod_{i=s}^{t-1} c_i
            #                 * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
            path_slice = slice(self.path_start_idx, self.ptr)

            obs = self.actor_critic.obs_oms(self.obs_buf[path_slice],
                                            clip=False) \
                if self.standardize_env_obs else self.obs_buf[path_slice]

            obs = torch.as_tensor(obs, dtype=torch.float32)

            act = self.act_buf[path_slice]
            act = torch.as_tensor(act, dtype=torch.float32)
            with torch.no_grad():
                # get current log_p of actions
                dist = self.actor_critic.pi.dist(obs)
                log_p = self.actor_critic.pi.log_prob_from_dist(dist, act)
            value_net_targets, adv, _ = calculate_v_trace(
                policy_action_probs=np.exp(log_p.numpy()),
                values=vals,
                rewards=rews,
                behavior_action_probs=np.exp(self.logp_buf[path_slice]),
                gamma=self.gamma,
                rho_bar=1.0,  # default is 1.0
                c_bar=1.0  # default is 1.0
            )

        elif self.adv_estimation_method == 'plain':
            # A(x, u) = Q(x, u) - V(x) = r(x, u) + gamma V(x+1) - V(x)
            adv = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

            # compute rewards-to-go, to be targets for the value function update
            # value_net_targets are just the discounted returns
            value_net_targets = discount_cumsum(rews, self.gamma)[:-1]

        else:
            raise NotImplementedError

        return adv, value_net_targets

    def store(self, obs, act, rew, val, logp, cost=0., cost_val=0.):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Important Note: Store only raw data received from environment!!!
        Note: perform reward scaling if enabled
        """
        assert self.ptr < self.max_size, f'No empty space in buffer'

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.cost_buf[self.ptr] = cost
        self.cost_val_buf[self.ptr] = cost_val
        self.ptr += 1

    def finish_path(self, last_val=0, last_cost_val=0, penalty_param=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cost_val)
        cost_vs = np.append(self.cost_val_buf[path_slice], last_cost_val)

        # new: add discounted returns to buffer
        discounted_ret = discount_cumsum(rews, self.gamma)[:-1]
        self.discounted_ret_buf[path_slice] = discounted_ret

        if self.use_reward_penalty:
            assert penalty_param >= 0, 'reward_penalty assumes positive value.'
            rews -= penalty_param * costs

        if self.use_scaled_rewards:
            # divide rewards by running return stddev.
            # discounted_ret = discount_cumsum(rews, self.gamma)[:-1]
            # for i, ret in enumerate(discounted_ret):
            # update running return statistics
            # self.actor_critic.ret_oms.update(discounted_ret)
            # # now scale...
            rews = self.actor_critic.ret_oms(rews, subtract_mean=False, clip=True)

        adv, v_targets = self.calculate_adv_and_value_targets(vals, rews)
        self.adv_buf[path_slice] = adv
        self.target_val_buf[path_slice] = v_targets

        # calculate costs
        c_adv, c_targets = self.calculate_adv_and_value_targets(cost_vs, costs,
                                                                lam=self.lam_c)
        self.cost_adv_buf[path_slice] = c_adv
        self.target_cost_val_buf[path_slice] = c_targets

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # TODO: pre-processing like standardization and scaling is done in
        #  Algorithm.  pre_process_data() method
        # if self.standardize_advantages:
        #     # the next two lines implement the advantage normalization trick
        #     adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        #     self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1.0e-8)
        #
        #     # also for cost advantages; only re-center but no rescale!
        #     self.cost_adv_buf = self.cost_adv_buf - np.mean(self.cost_adv_buf)

        # obs = self.actor_critic.obs_oms(self.obs_buf, clip=False) \
        #     if self.standardize_env_obs else self.obs_buf

        data = dict(
            obs=self.obs_buf, act=self.act_buf, target_v=self.target_val_buf,
            adv=self.adv_buf, log_p=self.logp_buf,
            # rew=self.rew_buf,
            discounted_ret=self.discounted_ret_buf,
            cost_adv=self.cost_adv_buf, target_c=self.target_cost_val_buf,
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in
                data.items()}
