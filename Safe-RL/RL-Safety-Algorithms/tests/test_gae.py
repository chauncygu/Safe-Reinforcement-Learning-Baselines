import unittest
import numpy as np
import gym
from rl_safety_algorithms.algs import core


def gae_baselines(values, rewards, last_val, gamma, lam):
    """ this is from OpenAI's baselines repository.
        see line 76 and following:
        https://github.com/openai/baselines/blob/master/baselines/trpo_mpi/trpo_mpi.py#L76
    """

    T = len(rewards)
    new = np.append(np.zeros_like(rewards), 0)
    vpred = np.append(values, last_val)

    adv = np.zeros(T, 'float32')
    gaelam = adv

    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rewards[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    value_targets = adv + values

    return adv, value_targets


class TestGAE(unittest.TestCase):

    @classmethod
    def create_gae_data(cls, values, rewards, last_val, gamma, lam, horizon):
        env = gym.make('MountainCarContinuous-v0')
        ac = core.get_actor(
            actor='mlp',
            observation_space=env.observation_space,
            action_space=env.action_space,
            use_scaled_rewards=False,
            use_shared_weights=False,
            ac_kwargs={
                'pi': {'hidden_sizes': (64, 64),
                       'activation': 'tanh'},
                'val': {'hidden_sizes': (64, 64),
                        'activation': 'tanh'}
            },
            use_standardized_obs=False
        )

        buf = core.Buffer(
            actor_critic=ac,
            obs_dim=env.observation_space.shape,
            act_dim=env.action_space.shape,
            size=horizon,
            gamma=gamma,
            lam=lam,
            adv_estimation_method='gae',
            use_scaled_rewards=False,
            standardize_env_obs=False,
            standardize_advantages=False,
        )

        # fill buffer with values
        for i in range(len(rewards)):
            x = np.zeros_like(env.observation_space.shape)
            action = env.action_space.sample()
            logp = 0.9  # can be anything
            buf.store(x, action, rewards[i], values[i], logp)

        buf.finish_path(last_val)
        data = buf.get()

        return data['adv'].numpy(), data['target_v'].numpy()

    def test_gae(self):
        """ test V-trace, compare recursive vs forward calculation"""
        horizon = 15
        rews = np.linspace(start=0.0, stop=1.0, num=horizon)
        vals = np.linspace(start=0.0, stop=1.0, num=horizon)
        bootstrap_val = 1.
        lam = 0.97
        gamma = 0.95

        adv, target_vals = self.create_gae_data(
            values=vals,
            rewards=rews,
            gamma=gamma,
            lam=lam,
            last_val=bootstrap_val,
            horizon=horizon
        )
        bs_adv, bs_target_vals = gae_baselines(
            values=vals,
            rewards=rews,
            gamma=gamma,
            lam=lam,
            last_val=bootstrap_val
        )

        self.assertTrue(np.allclose(adv, bs_adv))
        self.assertTrue(np.allclose(target_vals, bs_target_vals))


if __name__ == '__main__':
    unittest.main()
