import copy
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter

from macpo.utils.separated_buffer import SeparatedReplayBuffer
from macpo.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.gamma = self.all_args.gamma
        self.use_popart = self.all_args.use_popart

        self.safty_bound = self.all_args.safty_bound

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        from macpo.algorithms.r_mappo.r_macpo import R_MACTRPO_CPO as TrainAlgo

        from macpo.algorithms.r_mappo.algorithm.MACPPOPolicy import MACPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
            self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        # todo: revise this for trpo
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
            self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                  self.buffer[agent_id].rnn_states_critic[-1],
                                                                  self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

            next_costs = self.trainer[agent_id].policy.get_cost_values(self.buffer[agent_id].share_obs[-1],
                                                                       self.buffer[agent_id].rnn_states_cost[-1],
                                                                       self.buffer[agent_id].masks[-1])
            next_costs = _t2n(next_costs)
            self.buffer[agent_id].compute_cost_returns(next_costs, self.trainer[agent_id].value_normalizer)

    def train(self):
        # have modified for SAD_PPO
        train_infos = []
        cost_train_infos = []
        # random update order
        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, action_dim), dtype=np.float32)
        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[
                                                                                   agent_id].available_actions.shape[
                                                                               2:])

            if self.all_args.algorithm_name == "macpo":
                old_actions_logprob, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                    self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                    self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                    self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                    self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                    available_actions,
                    self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            # safe_buffer, cost_adv = self.buffer_filter(agent_id)
            # train_info = self.trainer[agent_id].train(safe_buffer, cost_adv)

            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, dist_entropy, action_mu, action_std = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            factor = factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(self.episode_length,
                                                                                                self.n_rollout_threads,
                                                                                                action_dim))
            train_infos.append(train_info)

            self.buffer[agent_id].after_update()

        return train_infos, cost_train_infos

    # episode length of envs is exactly equal to buffer size, that is, num_thread = num_episode
    def buffer_filter(self, agent_id):
        episode_length = len(self.buffer[0].rewards)
        # J constraints for all agents, just a toy example
        J = np.zeros((self.n_rollout_threads, 1), dtype=np.float32)
        for t in reversed(range(episode_length)):
            J = self.buffer[agent_id].costs[t] + self.gamma * J

        factor = self.buffer[agent_id].factor

        if self.use_popart:
            cost_adv = self.buffer[agent_id].cost_returns[:-1] - \
                       self.trainer[agent_id].value_normalizer.denormalize(self.buffer[agent_id].cost_preds[:-1])
        else:
            cost_adv = self.buffer[agent_id].cost_returns[:-1] - self.buffer[agent_id].cost_preds[:-1]

        expectation = np.mean(factor * cost_adv, axis=(0, 2))

        constraints_value = J + np.expand_dims(expectation, -1)

        del_id = []
        for i in range(self.n_rollout_threads):
            if constraints_value[i][0] > self.safty_bound:
                del_id.append(i)

        buffer_filterd = self.remove_episodes(agent_id, del_id)
        return buffer_filterd, cost_adv

    def remove_episodes(self, agent_id, del_ids):
        buffer = copy.deepcopy(self.buffer[agent_id])
        buffer.share_obs = np.delete(buffer.share_obs, del_ids, 1)
        buffer.obs = np.delete(buffer.obs, del_ids, 1)
        buffer.rnn_states = np.delete(buffer.rnn_states, del_ids, 1)
        buffer.rnn_states_critic = np.delete(buffer.rnn_states_critic, del_ids, 1)
        buffer.rnn_states_cost = np.delete(buffer.rnn_states_cost, del_ids, 1)
        buffer.value_preds = np.delete(buffer.value_preds, del_ids, 1)
        buffer.returns = np.delete(buffer.returns, del_ids, 1)
        if buffer.available_actions is not None:
            buffer.available_actions = np.delete(buffer.available_actions, del_ids, 1)
        buffer.actions = np.delete(buffer.actions, del_ids, 1)
        buffer.action_log_probs = np.delete(buffer.action_log_probs, del_ids, 1)
        buffer.rewards = np.delete(buffer.rewards, del_ids, 1)
        # todo: cost should be calculated entirely
        buffer.costs = np.delete(buffer.costs, del_ids, 1)
        buffer.cost_preds = np.delete(buffer.cost_preds, del_ids, 1)
        buffer.cost_returns = np.delete(buffer.cost_returns, del_ids, 1)
        buffer.masks = np.delete(buffer.masks, del_ids, 1)
        buffer.bad_masks = np.delete(buffer.bad_masks, del_ids, 1)
        buffer.active_masks = np.delete(buffer.active_masks, del_ids, 1)
        if buffer.factor is not None:
            buffer.factor = np.delete(buffer.factor, del_ids, 1)
        return buffer

    def save(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
