""" PyTorch implementation of Lagrangian Trust Region Policy Optimization

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    14.10.2020
Updated:    20.10.2020
"""
import numpy as np
from torch import optim
import torch
from rl_safety_algorithms.common import utils
from rl_safety_algorithms.algs.core import ConstrainedPolicyGradientAlgorithm
from rl_safety_algorithms.algs.trpo.trpo import TRPOAlgorithm
import rl_safety_algorithms.common.mpi_tools as mpi_tools


class LagrangianTRPOAlgorithm(TRPOAlgorithm,
                              ConstrainedPolicyGradientAlgorithm):
    def __init__(
            self,
            alg: str = 'lag-trpo',
            cost_limit: float = 25.,
            lagrangian_multiplier_init: float = 0.001,
            lambda_lr: float = 0.05,
            lambda_optimizer: str = 'SGD',
            use_lagrangian_penalty: bool = True,
            **kwargs
    ):
        TRPOAlgorithm.__init__(
            self,
            alg=alg,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            use_cost_value_function=True,
            use_kl_early_stopping=False,
            use_lagrangian_penalty=use_lagrangian_penalty,
            **kwargs
        )

        ConstrainedPolicyGradientAlgorithm.__init__(
            self,
            cost_limit=cost_limit,
            use_lagrangian_penalty=use_lagrangian_penalty,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer
        )

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict) -> tuple:
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        if self.use_lagrangian_penalty:
            # ensure that lagrange multiplier is positive
            penalty = self.lambda_range_projection(self.lagrangian_multiplier)
            loss_pi += penalty * (ratio * data['cost_adv']).mean()
            loss_pi /= (1 + penalty)

        # Useful extra info
        approx_kl = .5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)


def get_alg(env_id, **kwargs) -> LagrangianTRPOAlgorithm:
    return LagrangianTRPOAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='lag-trpo', env_id=env_id)
    defaults.update(**kwargs)

    alg = LagrangianTRPOAlgorithm(
        env_id=env_id,
        **defaults
    )

    ac, env = alg.learn()

    return ac, env
