import numpy as np
import torch
import torch.nn as nn
from macpo.utils.util import get_gard_norm, huber_loss, mse_loss
from macpo.utils.popart import PopArt
from macpo.algorithms.utils.util import check
from macpo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from torch.nn.utils import clip_grad_norm
import copy


# EPS = 1e-8

class R_MACTRPO_CPO():
    """
    Trainer class for MATRPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy, attempt_feasible_recovery=False,
                 attempt_infeasible_recovery=False, revert_to_last_safe_point=False, delta_bound=0.011,
                 safety_bound=0.1,
                 _backtrack_ratio=0.8, _max_backtracks=15, _constraint_name_1="trust_region",
                 _constraint_name_2="safety_region", linesearch_infeasible_recovery=True, accept_violation=False,
                 learn_margin=False,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.episode_length = args.episode_length

        self.kl_threshold = args.kl_threshold
        self.safety_bound = args.safety_bound
        self.ls_step = args.ls_step
        self.accept_ratio = args.accept_ratio
        self.EPS = args.EPS
        self.gamma = args.gamma
        self.safety_gamma = args.safety_gamma
        self.line_search_fraction = args.line_search_fraction
        self.g_step_dir_coef = args.g_step_dir_coef
        self.b_step_dir_coef = args.b_step_dir_coef
        self.fraction_coef = args.fraction_coef

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # todo:  my args-start
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self._damping = 0.0001
        self._delta = 0.01
        self._max_backtracks = 10
        self._backtrack_coeff = 0.5

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.attempt_feasible_recovery = attempt_feasible_recovery
        self.attempt_infeasible_recovery = attempt_infeasible_recovery
        self.revert_to_last_safe_point = revert_to_last_safe_point
        self._max_quad_constraint_val = args.kl_threshold  # delta_bound
        self._max_lin_constraint_val = args.safety_bound
        self._backtrack_ratio = _backtrack_ratio
        self._max_backtracks = _max_backtracks
        self._constraint_name_1 = _constraint_name_1
        self._constraint_name_2 = _constraint_name_2
        self._linesearch_infeasible_recovery = linesearch_infeasible_recovery
        self._accept_violation = accept_violation

        hvp_approach = None
        num_slices = 1
        self.lamda_coef = 0
        self.lamda_coef_a_star = 0
        self.lamda_coef_b_star = 0

        self.margin = 0
        self.margin_lr = 0.05
        self.learn_margin = learn_margin
        self.n_rollout_threads = args.n_rollout_threads


        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    def kl_divergence(self, obs, rnn_states, action, masks, available_actions, active_masks, new_actor, old_actor):

        _, _, mu, std = new_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
        _, _, mu_old, std_old = old_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions,
                                                           active_masks)
        logstd = torch.log(std)
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        logstd_old = torch.log(std_old)

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
        # be careful of calculating KL-divergence. It is not symmetric metric
        kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
             (self.EPS + 2.0 * std.pow(2)) - 0.5

        return kl.sum(1, keepdim=True)

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def conjugate_gradient(self, actor, obs, rnn_states, action, masks, available_actions, active_masks, b, nsteps,
                           residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device=self.device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(actor, obs, rnn_states, action, masks, available_actions, active_masks, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def fisher_vector_product(self, actor, obs, rnn_states, action, masks, available_actions, active_masks, p):
        p.detach()
        kl = self.kl_divergence(obs, rnn_states, action, masks, available_actions, active_masks, new_actor=actor,
                                old_actor=actor)
        kl = kl.mean()
        kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True, allow_unused=True)
        kl_grad = self.flat_grad(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters(), allow_unused=True)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    def _get_flat_grad(self, y, model, retain_graph=None, create_graph=False):
        grads = torch.autograd.grad(y, model.parameters(), retain_graph=retain_graph,
                                    create_graph=create_graph, allow_unused=True)
        _grads = []
        for val, p in zip(grads, model.parameters()):
            if val is not None:
                _grads.append(val)
            else:
                _grads.append(torch.zeros_like(p.data, requires_grad=create_graph))
        return torch.cat([grad.reshape(-1) for grad in _grads])

    def _flat_grad_(self, f, model, retain_graph=None, create_graph=False):
        return self.flat_grad(torch.autograd.grad(f, model.parameters(), retain_graph=retain_graph,
                                                  create_graph=create_graph, allow_unused=True))

    def hessian_vector_product(self, f, model):
        # for H = grad**2 f, compute Hx
        g = self._flat_grad_(f, model)
        # g = self._get_flat_grad(f, model)
        # x = torch.placeholder(torch.float32, shape=g.shape)
        x = torch.FloatTensor(g.shape)
        return x, self._flat_grad_(torch.sum(g * x), model)

    def cg(self, Ax, b, cg_iters=10):
        x = np.zeros_like(b)
        r = b.clone()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (torch.dot(p, z) + self.EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def trpo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_returns_barch, rnn_states_cost_batch, \
        cost_adv_targ, aver_episode_costs = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        cost_returns_barch = check(cost_returns_barch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)
        # Reshape to do in a single forward pass for all steps
        # values, action_log_probs, dist_entropy, cost_values, action_mu, action_std

        values, action_log_probs, dist_entropy, cost_values, action_mu, action_std = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            rnn_states_cost_batch)

        # todo: reward critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        # todo: cost critic update
        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_barch, active_masks_batch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.max_grad_norm)
        else:
            cost_grad_norm = get_gard_norm(self.policy.cost_critic.parameters())
        self.policy.cost_optimizer.step()

        # todo: actor update

        rescale_constraint_val = (aver_episode_costs.mean() - self._max_lin_constraint_val) * (1 - self.gamma)

        if rescale_constraint_val == 0:
            rescale_constraint_val = self.EPS

        # todo:reward-g
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        if self._use_policy_active_masks:
            reward_loss = (torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True) *
                           active_masks_batch).sum() / active_masks_batch.sum()
        else:
            reward_loss = torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True).mean()
        reward_loss = - reward_loss  # todo:
        reward_loss_grad = torch.autograd.grad(reward_loss, self.policy.actor.parameters(), retain_graph=True,
                                               allow_unused=True)
        reward_loss_grad = self.flat_grad(reward_loss_grad)

        # todo:cost-b
        if self._use_policy_active_masks:
            cost_loss = (torch.sum(ratio * factor_batch * (cost_adv_targ), dim=-1, keepdim=True) *
                         active_masks_batch).sum() / active_masks_batch.sum()
        else:
            cost_loss = torch.sum(ratio * factor_batch * (cost_adv_targ), dim=-1, keepdim=True).mean()
        cost_loss_grad = torch.autograd.grad(cost_loss, self.policy.actor.parameters(), retain_graph=True,
                                             allow_unused=True)
        cost_loss_grad = self.flat_grad(cost_loss_grad)
        B_cost_loss_grad = cost_loss_grad.unsqueeze(0)
        B_cost_loss_grad = self.flat_grad(B_cost_loss_grad)

        # todo: compute lamda_coef and v_coef
        g_step_dir = self.conjugate_gradient(self.policy.actor,
                                             obs_batch,
                                             rnn_states_batch,
                                             actions_batch,
                                             masks_batch,
                                             available_actions_batch,
                                             active_masks_batch,
                                             reward_loss_grad.data,
                                             nsteps=10)  # todo: compute H^{-1} g
        b_step_dir = self.conjugate_gradient(self.policy.actor,
                                             obs_batch,
                                             rnn_states_batch,
                                             actions_batch,
                                             masks_batch,
                                             available_actions_batch,
                                             active_masks_batch,
                                             B_cost_loss_grad.data,
                                             nsteps=10)  # todo: compute H^{-1} b

        q_coef = (reward_loss_grad * g_step_dir).sum(0, keepdim=True)  # todo: compute q_coef: = g^T H^{-1} g
        r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)  # todo: compute r_coef: = g^T H^{-1} b
        s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)  # todo: compute s_coef: = b^T H^{-1} b

        fraction = self.line_search_fraction #0.5 # 0.5  # line search step size
        loss_improve = 0  # initialization

        """self._max_lin_constraint_val = c, B_cost_loss_grad = c in cpo"""

        B_cost_loss_grad_dot = torch.dot(B_cost_loss_grad, B_cost_loss_grad)
        # torch.dot(B_cost_loss_grad, B_cost_loss_grad) # B_cost_loss_grad.mean() * B_cost_loss_grad.mean()
        if (torch.dot(B_cost_loss_grad, B_cost_loss_grad)) <= self.EPS and rescale_constraint_val < 0:
            # feasible and cost grad is zero---shortcut to pure TRPO update!
            # w, r, s, A, B = 0, 0, 0, 0, 0
            # g_step_dir = torch.tensor(0)
            b_step_dir = torch.tensor(0)
            r_coef = torch.tensor(0)
            s_coef = torch.tensor(0)
            positive_Cauchy_value = torch.tensor(0)
            whether_recover_policy_value = torch.tensor(0)
            optim_case = 4
            # print("optim_case = 4---shortcut to pure TRPO update!")
        else:
            # cost grad is nonzero: CPO update!
            r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)  # todo: compute r_coef: = g^T H^{-1} b
            s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)  # todo: compute s_coef: = b^T H^{-1} b
            if r_coef == 0:
                r_coef = self.EPS
            if s_coef == 0:
                s_coef = self.EPS
            positive_Cauchy_value = (
                        q_coef - (r_coef ** 2) / (self.EPS + s_coef))  # should be always positive (Cauchy-Shwarz)
            whether_recover_policy_value = 2 * self._max_quad_constraint_val - (
                    rescale_constraint_val ** 2) / (
                                                       self.EPS + s_coef)  # does safety boundary intersect trust region? (positive = yes)
            if rescale_constraint_val < 0 and whether_recover_policy_value < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
                # print("optim_case = 3---entire trust region is feasible")
            elif rescale_constraint_val < 0 and whether_recover_policy_value >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
                # print('optim_case = 2---most of trust region is feasible')
            elif rescale_constraint_val >= 0 and whether_recover_policy_value >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                # print('optim_case = 1---Alert! Attempting feasible recovery!')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                # print('optim_case = 0---Alert! Attempting infeasible recovery!')
        if whether_recover_policy_value == 0:
            whether_recover_policy_value = self.EPS
        
        if optim_case in [3, 4]:
            lam = torch.sqrt(
                (q_coef / (2 * self._max_quad_constraint_val)))  # self.lamda_coef = lam = np.sqrt(q / (2 * target_kl))
            nu = torch.tensor(0)  # v_coef = 0
        elif optim_case in [1, 2]:
            LA, LB = [0, r_coef / rescale_constraint_val], [r_coef / rescale_constraint_val, np.inf]
            LA, LB = (LA, LB) if rescale_constraint_val < 0 else (LB, LA)
            proj = lambda x, L: max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(positive_Cauchy_value / whether_recover_policy_value), LA)
            lam_b = proj(torch.sqrt(q_coef / (torch.tensor(2 * self._max_quad_constraint_val))), LB)

            f_a = lambda lam: -0.5 * (positive_Cauchy_value / (
                        self.EPS + lam) + whether_recover_policy_value * lam) - r_coef * rescale_constraint_val / (
                                          self.EPS + s_coef)
            f_b = lambda lam: -0.5 * (q_coef / (self.EPS + lam) + 2 * self._max_quad_constraint_val * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * rescale_constraint_val - r_coef) / (self.EPS + s_coef)
        else:
            lam = torch.tensor(0)
            nu = torch.sqrt(torch.tensor(2 * self._max_quad_constraint_val) / (self.EPS + s_coef))

        x_a = (1. / (lam + self.EPS)) * (g_step_dir + nu * b_step_dir)
        x_b = (nu * b_step_dir)
        x = x_a if optim_case > 0 else x_b

        # todo: update actor and learning
        reward_loss = reward_loss.data.cpu().numpy()
        cost_loss = cost_loss.data.cpu().numpy()
        params = self.flat_params(self.policy.actor)

        old_actor = R_Actor(self.policy.args,
                            self.policy.obs_space,
                            self.policy.act_space,
                            self.device)
        self.update_model(old_actor, params)

        expected_improve = -torch.dot(x, reward_loss_grad).sum(0, keepdim=True)
        expected_improve = expected_improve.data.cpu().numpy()

        # line search
        flag = False
        fraction_coef = self.fraction_coef
        # print("fraction_coef", fraction_coef)
        for i in range(self.ls_step):
            x_norm = torch.norm(x)
            if x_norm > 0.5:
                x = x * 0.5 / x_norm

            new_params = params - fraction_coef * (fraction**i) * x
            self.update_model(self.policy.actor, new_params)
            values, action_log_probs, dist_entropy, new_cost_values, action_mu, action_std = self.policy.evaluate_actions(
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                available_actions_batch,
                active_masks_batch,
                rnn_states_cost_batch)

            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            if self._use_policy_active_masks:
                new_reward_loss = (torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True) *
                                   active_masks_batch).sum() / active_masks_batch.sum()
            else:
                new_reward_loss = torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True).mean()

            if self._use_policy_active_masks:
                new_cost_loss = (torch.sum(ratio * factor_batch * cost_adv_targ, dim=-1, keepdim=True) *
                                 active_masks_batch).sum() / active_masks_batch.sum()
            else:
                new_cost_loss = torch.sum(ratio * factor_batch * cost_adv_targ, dim=-1, keepdim=True).mean()

            new_reward_loss = new_reward_loss.data.cpu().numpy()
            new_reward_loss = -new_reward_loss
            new_cost_loss = new_cost_loss.data.cpu().numpy()
            loss_improve = new_reward_loss - reward_loss

            kl = self.kl_divergence(obs_batch,
                                    rnn_states_batch,
                                    actions_batch,
                                    masks_batch,
                                    available_actions_batch,
                                    active_masks_batch,
                                    new_actor=self.policy.actor,
                                    old_actor=old_actor)
            kl = kl.mean()

            # see https: // en.wikipedia.org / wiki / Backtracking_line_search
            if ((kl < self.kl_threshold) and (loss_improve < 0 if optim_case > 1 else True)
                    and (new_cost_loss.mean() - cost_loss.mean() <= max(-rescale_constraint_val, 0))):
                flag = True
                # print("line search successful")
                break
            expected_improve *= fraction

        if not flag:
            # line search failed
            print("line search failed")
            params = self.flat_params(old_actor)
            self.update_model(self.policy.actor, params)

        return value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, ratio, cost_loss, cost_grad_norm, whether_recover_policy_value, cost_preds_batch, cost_returns_barch, B_cost_loss_grad, lam, nu, g_step_dir, b_step_dir, x, action_mu, action_std, B_cost_loss_grad_dot

    def train(self, buffer, shared_buffer=None, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self._use_popart:
            cost_adv = buffer.cost_returns[:-1] - self.value_normalizer.denormalize(buffer.cost_preds[:-1])
        else:
            cost_adv = buffer.cost_returns[:-1] - buffer.cost_preds[:-1]
        cost_adv_copy = cost_adv.copy()
        cost_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_adv = np.nanmean(cost_adv_copy)
        std_cost_adv = np.nanstd(cost_adv_copy)
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['kl'] = 0
        train_info['dist_entropy'] = 0
        train_info['loss_improve'] = 0
        train_info['expected_improve'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cost_loss'] = 0
        train_info['cost_grad_norm'] = 0
        train_info['whether_recover_policy_value'] = 0
        train_info['cost_preds_batch'] = 0
        train_info['cost_returns_barch'] = 0
        train_info['B_cost_loss_grad'] = 0
        train_info['lam'] = 0
        train_info['nu'] = 0
        train_info['g_step_dir'] = 0
        train_info['b_step_dir'] = 0
        train_info['x'] = 0
        train_info['action_mu'] = 0
        train_info['action_std'] = 0
        train_info['B_cost_loss_grad_dot'] = 0

        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length,
                                                        cost_adv=cost_adv)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch, cost_adv=cost_adv)
        else:
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, cost_adv=cost_adv)
        # old_actor = copy.deepcopy(self.policy.actor)
        for sample in data_generator:
            value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, imp_weights, cost_loss, cost_grad_norm, whether_recover_policy_value, cost_preds_batch, cost_returns_barch, B_cost_loss_grad, lam, nu, g_step_dir, b_step_dir, x, action_mu, action_std, B_cost_loss_grad_dot \
                = self.trpo_update(sample, update_actor)

            train_info['value_loss'] += value_loss.item()
            train_info['kl'] += kl
            train_info['loss_improve'] += loss_improve
            train_info['expected_improve'] += expected_improve
            train_info['dist_entropy'] += dist_entropy.item()
            train_info['critic_grad_norm'] += critic_grad_norm
            train_info['ratio'] += imp_weights.mean()
            train_info['cost_loss'] += value_loss.item()
            train_info['cost_grad_norm'] += cost_grad_norm
            train_info['whether_recover_policy_value'] += whether_recover_policy_value
            train_info['cost_preds_batch'] += cost_preds_batch.mean()
            train_info['cost_returns_barch'] += cost_returns_barch.mean()
            train_info['B_cost_loss_grad'] += B_cost_loss_grad.mean()

            train_info['g_step_dir'] += g_step_dir.float().mean()
            train_info['b_step_dir'] += b_step_dir.float().mean()
            train_info['x'] = x.float().mean()
            train_info['action_mu'] += action_mu.float().mean()
            train_info['action_std'] += action_std.float().mean()
            train_info['B_cost_loss_grad_dot'] += B_cost_loss_grad_dot.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    """
    B_cost_loss_grad_dot = torch.dot(B_cost_loss_grad, B_cost_loss_grad)
    if torch.dot(B_cost_loss_grad, B_cost_loss_grad) <= 1e-8 and rescale_constraint_val < 0:
        b_step_dir, r_coef, s_coef, A, B = 0, 0, 0, 0, 0
        optim_case = 4

    else:
        A = q_coef - r_coef**2/s_coef
        B = self._max_quad_constraint_val - (rescale_constraint_val ** 2) / (s_coef+ self.EPS)
        positive_Cauchy_value = A
        whether_recover_policy_value = B
        if rescale_constraint_val<0 and B<0:
            optim_case = 3

        elif rescale_constraint_val < 0 and B >= 0:
            optim_case = 2

        elif rescale_constraint_val >= 0 and B >= 0:
            optim_case = 1

        else:
            optim_case = 0
        if A==0:
            A = self.EPS
        if B==0:
            B = self.EPS

    lam, nu = 0, 0
    if optim_case == 0:  # need to recover policy from unfeasible point
        recover_policy_flag = True
        lam = 0
        nu = torch.sqrt(2 * self.kl_threshold / (s_coef + self.EPS) )

    elif optim_case in [1, 2]:
        lamda_a = torch.sqrt(A/B)
        lamda_A_1 = r_coef / rescale_constraint_val
        lamda_A_2 = torch.tensor(0)
        lamda_b = torch.sqrt(q_coef / (2 * self._max_quad_constraint_val))
        if rescale_constraint_val > 0:
            lamda_coef_1 = torch.max(lamda_A_1, lamda_a)  # assume lamda*c - r >0
            lamda_coef_2 = torch.max(lamda_A_2, torch.min(lamda_b, lamda_A_1))  # assume lamda*c - r < 0
            if (lamda_coef_1 * rescale_constraint_val - r_coef) > 0:  # assume lamda*c - r >0 successfully
                self.lamda_coef_a_star = lamda_coef_1
            else:  # assume failed
                self.lamda_coef_b_star = lamda_coef_2
        else:
            lamda_coef_3 = torch.max(lamda_A_2, torch.min(lamda_a, lamda_A_1))  # assume lamda*c - r >0
            lamda_coef_4 = torch.max(lamda_b, lamda_A_1)  # assume lamda*c - r < 0
            # print("lamda_coef_3 * rescale_constraint_val - r_coef ",
            # lamda_coef_3 * rescale_constraint_val - r_coef)
            if lamda_coef_3 * rescale_constraint_val - r_coef > 0:
                self.lamda_coef_a_star = lamda_coef_3
            else:
                self.lamda_coef_b_star = lamda_coef_4
        if self.lamda_coef_b_star==0:
            self.lamda_coef_b_star = self.EPS
        if self.lamda_coef_a_star==0:
            self.lamda_coef_a_star = self.EPS
        if s_coef==0:
            s_coef = self.EPS
        f_a_star = -A/(2*self.lamda_coef_a_star +  self.EPS) - self.lamda_coef_a_star*B/2 - r_coef*rescale_constraint_val/(s_coef+ self.EPS)
        f_b_star = -(self._max_quad_constraint_val/(self.lamda_coef_b_star+ self.EPS) \
                    + self.lamda_coef_b_star*self._max_quad_constraint_val)/2

        if f_a_star > f_b_star:
            lam = self.lamda_coef_a_star
        else:
            lam = self.lamda_coef_b_star

        nu = torch.relu( (lam*rescale_constraint_val - r_coef)/(s_coef + self.EPS) )

    elif optim_case in [3, 4]:
        lam = torch.sqrt(q_coef/(2*self._max_quad_constraint_val))
        nu = 0.
    """

