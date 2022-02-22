""" PyTorch implementation of the Primal Dual Optimization (PDO) algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    28.10.2020
Updated:    --

inspired by:
    Joshua Achiam, David Held, Aviv Tamar, Peter Abbeel
    Constrained Policy Optimization
    ICML 2017

also see:
    Yinlam Chow, Mohammad Ghavamzadeh, Lucas Janson, and Marco Pavone
    Risk-constrained reinforcement learning with percentile risk criteria
    J. Mach. Learn. Res. 2017
"""
import numpy as np
from torch import optim
import torch
from rl_safety_algorithms.algs.cpo.cpo import CPOAlgorithm
from rl_safety_algorithms.algs.core import ConstrainedPolicyGradientAlgorithm
from rl_safety_algorithms.algs.npg.npg import NaturalPolicyGradientAlgorithm
from rl_safety_algorithms.algs.trpo.trpo import TRPOAlgorithm
import rl_safety_algorithms.algs.utils as U
from rl_safety_algorithms.common import utils
import rl_safety_algorithms.common.mpi_tools as mpi_tools


class PrimalDualOptimizationAlgorithm(CPOAlgorithm,
                                      ConstrainedPolicyGradientAlgorithm):
    def __init__(
            self,
            alg: str = 'pdo',
            cost_limit: float = 25.,
            lagrangian_multiplier_init: float = 0.001,
            lambda_optimizer: str = 'Adam',
            lambda_lr: float = 0.001,
            **kwargs
    ):
        CPOAlgorithm.__init__(
            self,
            alg=alg,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            **kwargs
        )
        assert self.alg == 'pdo'  # sanity check of argument passing

        ConstrainedPolicyGradientAlgorithm.__init__(
            self,
            cost_limit=cost_limit,
            use_lagrangian_penalty=True,  # todo: this param might be of no relevance
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer
        )

    def algorithm_specific_logs(self):
        NaturalPolicyGradientAlgorithm.algorithm_specific_logs(self)
        self.logger.log_tabular('Misc/cost_gradient_norm')
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def update(self):
        # First update Lagrange multiplier parameter
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        super().update()

    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = U.get_flat_params_from(self.ac.pi.net)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        self.loss_v_before = self.compute_loss_v(data['obs'],
                                                 data['target_v']).item()
        self.loss_c_before = self.compute_loss_c(data['obs'],
                                                 data['target_c']).item()
        # get prob. distribution before updates
        p_dist = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = U.get_flat_gradients_from(self.ac.pi.net)
        g_flat *= -1  # flip sign since policy_loss = -(ration * adv)

        # get the policy cost performance gradient b (flat as vector)
        self.pi_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(data=data)
        loss_cost.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        self.loss_pi_cost_before = loss_cost.item()
        b_flat = U.get_flat_gradients_from(self.ac.pi.net)

        p = g_flat - self.lagrangian_multiplier * b_flat

        x = U.conjugate_gradients(self.Fvp, p, self.cg_iters)
        assert torch.isfinite(x).all()
        pHp = torch.dot(x, self.Fvp(x))  # equivalent to : p^T x
        assert pHp.item() >= 0, 'No negative values.'

        # perform descent direction
        eps = 1.0e-8
        alpha = torch.sqrt(2 * self.target_kl / (pHp + eps))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()
        ep_costs = self.logger.get_stats('EpCosts')[0]
        c = ep_costs - self.cost_limit
        c /= (self.logger.get_stats('EpLen')[0] + eps)  # rescale

        # determine step direction and apply SGD step after grads where set
        final_step_dir, accept_step = self.adjust_step_direction(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            data=data
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        U.set_param_values_to_model(self.ac.pi.net, new_theta)

        with torch.no_grad():
            q_dist = self.ac.pi.dist(data['obs'])
            kl = torch.distributions.kl.kl_divergence(p_dist,
                                                      q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.logger.store(**{
            'Values/Adv': data['act'].numpy(),
            'Entropy': pi_info['ent'],
            'KL': kl,
            'PolicyRatio': pi_info['ratio'],
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/AcceptanceStep': accept_step,
            'Misc/Alpha': alpha.item(),
            'Misc/StopIter': 1,
            'Misc/FinalStepNorm': final_step_dir.norm().item(),
            'Misc/xHx': pHp.item(),
            'Misc/gradient_norm': torch.norm(g_flat).item(),
            'Misc/cost_gradient_norm': torch.norm(b_flat).item(),
            'Misc/H_inv_g': x.norm().item(),
        })


def get_alg(env_id, **kwargs) -> PrimalDualOptimizationAlgorithm:
    return PrimalDualOptimizationAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(env_id,
          **kwargs
          ) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='pdo', env_id=env_id)
    defaults.update(**kwargs)
    alg = PrimalDualOptimizationAlgorithm(
        env_id=env_id,
        **defaults
    )
    ac, env = alg.learn()
    return ac, env
