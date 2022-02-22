import torch
from rl_safety_algorithms.algs.npg.npg import NaturalPolicyGradientAlgorithm
import rl_safety_algorithms.algs.utils as U
from rl_safety_algorithms.common import utils
import rl_safety_algorithms.common.mpi_tools as mpi_tools


class TRPOAlgorithm(NaturalPolicyGradientAlgorithm):
    def __init__(
            self,
            alg: str = 'trpo',
            **kwargs
    ):
        super().__init__(alg=alg, **kwargs)

    def adjust_step_direction(self,
                              step_dir,
                              g_flat,
                              p_dist,
                              data,
                              total_steps: int = 15,
                              decay: float = 0.8
                              ) -> tuple:
        """ TRPO performs line-search until constraint satisfaction."""
        step_frac = 1.0
        _theta_old = U.get_flat_params_from(self.ac.pi.net)
        expected_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            U.set_param_values_to_model(self.ac.pi.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, pi_info = self.compute_loss_pi(data=data)
                # determine KL div between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            loss_improve = self.loss_pi_before - loss_pi.item()
            # average processes....
            torch_kl = mpi_tools.mpi_avg(torch_kl)
            loss_improve = mpi_tools.mpi_avg(loss_improve)

            self.logger.log("Expected Improvement: %.3f Actual: %.3f" % (
                expected_improve, loss_improve))
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        U.set_param_values_to_model(self.ac.pi.net, _theta_old)

        return step_frac * step_dir, acceptance_step


def get_alg(env_id, **kwargs) -> TRPOAlgorithm:
    return TRPOAlgorithm(
        env_id=env_id,
        **kwargs
    )


def learn(
        env_id,
        **kwargs
) -> tuple:
    defaults = utils.get_defaults_kwargs(alg='trpo', env_id=env_id)
    defaults.update(**kwargs)
    alg = TRPOAlgorithm(
        env_id=env_id,
        **defaults
    )

    ac, env = alg.learn()

    return ac, env
