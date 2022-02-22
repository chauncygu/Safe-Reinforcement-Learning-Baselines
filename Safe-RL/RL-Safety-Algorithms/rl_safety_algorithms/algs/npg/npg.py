""" PyTorch implementation of Natural Policy Gradient (NPG) Algorithm.

Author:     Sven Gronauer (sven.gronauer@tum.de)
Created:    10.10.2020
Updated:    21.11.2020
"""
import torch
from rl_safety_algorithms.algs.iwpg import iwpg
import rl_safety_algorithms.algs.utils as U
from rl_safety_algorithms.common import utils
import rl_safety_algorithms.common.mpi_tools as mpi_tools
import numpy as np


class NaturalPolicyGradientAlgorithm(iwpg.IWPGAlgorithm):
    def __init__(
            self,
            alg: str = 'npg',
            cg_damping: float = 0.1,
            cg_iters: int = 10,
            target_kl: float = 0.01,
            **kwargs
    ):
        super().__init__(
            alg=alg,
            cg_damping=cg_damping,
            cg_iters=cg_iters,
            target_kl=target_kl,
            **kwargs)
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.target_kl = target_kl
        self.fvp_obs = None
        self.scheduler = None  # disable scheduler if activated by parent class

    def adjust_step_direction(self,
                              step_dir,
                              g_flat,
                              p_dist,
                              data):
        """ NPG does not perform line search. This method is over-written by
            TRPOs backtracking line search."""
        accept_step = 1
        return step_dir, accept_step

    def algorithm_specific_logs(self):
        self.logger.log_tabular('Misc/AcceptanceStep')
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/FinalStepNorm')
        self.logger.log_tabular('Misc/gradient_norm')
        self.logger.log_tabular('Misc/xHx')
        self.logger.log_tabular('Misc/H_inv_g')

    def Fvp(self, p):
        """ Build the Hessian-vector product based on an approximation of the
            KL-divergence.

            For details see John Schulman's PhD thesis (pp. 40)
            http://joschu.net/docs/thesis.pdf
        """
        self.ac.pi.net.zero_grad()
        q_dist = self.ac.pi.dist(self.fvp_obs)
        with torch.no_grad():
            p_dist = self.ac.pi.dist(self.fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.ac.pi.net.parameters(),
                                    create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * p).sum()
        grads = torch.autograd.grad(kl_p, self.ac.pi.net.parameters(),
                                    retain_graph=False)
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                       for grad in grads])
        # average --->
        mpi_tools.mpi_avg_torch_tensor(flat_grad_grad_kl)
        return flat_grad_grad_kl + p * self.cg_damping

    def update(self) -> None:
        """Update value and policy networks. Note that the order doesn't matter.

        Returns
        -------
            None
        """
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Update Policy Network
        self.update_policy_net(data)
        # Update Value Function
        self.update_value_net(data=data)
        if self.use_cost_value_function:
            self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = U.get_flat_params_from(self.ac.pi.net)
        self.ac.pi.net.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = mpi_tools.mpi_avg(loss_pi.item())
        loss_v = self.compute_loss_v(data['obs'], data['target_v'])
        self.loss_v_before = mpi_tools.mpi_avg(loss_v.item())
        p_dist = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        mpi_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = U.get_flat_gradients_from(self.ac.pi.net)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = U.conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.adjust_step_direction(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            data=data,
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
            'Misc/FinalStepNorm': torch.norm(final_step_dir).numpy(),
            'Misc/xHx': xHx.item(),
            'Misc/gradient_norm': torch.norm(g_flat).numpy(),
            'Misc/H_inv_g': x.norm().item(),
        })


def get_alg(env_id, **kwargs) -> NaturalPolicyGradientAlgorithm:
    return NaturalPolicyGradientAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='npg', env_id=env_id)
    defaults.update(**kwargs)
    alg = NaturalPolicyGradientAlgorithm(
        env_id=env_id,
        **defaults
    )
    ac, env = alg.learn()

    return ac, env
