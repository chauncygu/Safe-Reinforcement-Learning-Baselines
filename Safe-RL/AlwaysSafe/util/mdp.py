from typing import Sequence
from tqdm import trange

import numpy as np
from gym_factored.envs.base import DiscreteEnv


def get_mdp_functions(env: DiscreteEnv):
    transition = np.zeros(shape=(env.nS, env.nA, env.nS))
    reward = np.zeros(shape=(env.nS, env.nA))
    cost = np.zeros(shape=(env.nS, env.nA))
    terminal = np.zeros(shape=env.nS, dtype=bool)
    for s, state_transitions in env.P.items():
        for a, state_action_transitions in state_transitions.items():
            for tr in state_action_transitions:
                p, ns, r, done, info = get_transition_with_info(tr)
                reward[s, a] += p * r
                cost[s, a] += p * info.get('cost', 0)
                transition[s, a, ns] = p
                if done:
                    terminal[ns] = True
    return transition, reward, cost, terminal


def get_transition_with_info(tr: Sequence):
    if len(tr) == 4:
        return (*tr, {})
    else:
        return (*tr,)


def get_mdp_functions_partial(env: DiscreteEnv, features: Sequence):
    """
    extracts an abstraction of the MDP that only considers the given features
    """

    feature_values = [set() for _ in features]
    for s in range(env.nS):
        decoded_state = list(env.decode(s))
        for i, feature in enumerate(features):
            feature_values[i].add(decoded_state[feature])
    feature_domains = []
    for i in range(len(features)):
        feature_domains.append(sorted(feature_values[i]))

    number_of_abstract_states = 1
    for feature_domain in feature_domains:
        number_of_abstract_states *= len(feature_domain)
    w = number_of_abstract_states / env.nS

    transition = np.zeros(shape=(number_of_abstract_states, env.nA, number_of_abstract_states))
    reward = np.zeros(shape=(number_of_abstract_states, env.nA))
    cost = np.zeros(shape=(number_of_abstract_states, env.nA))
    terminal = np.ones(shape=number_of_abstract_states, dtype=bool)
    abs_map = np.zeros(shape=(number_of_abstract_states, env.nS), dtype=bool)
    for s, state_transitions in env.P.items():
        abstract_s = abstract(s, features, feature_domains, env)
        if env.isd[s] > 0:
            terminal[abstract_s] = False
        abs_map[abstract_s, s] = 1
        for a, state_action_transitions in state_transitions.items():
            for tr in state_action_transitions:
                p, ns, r, done, info = get_transition_with_info(tr)
                abstract_ns = abstract(ns, features, feature_domains, env)
                reward[abstract_s, a] += p * r * w
                cost[abstract_s, a] += p * info.get('cost', 0) * w
                transition[abstract_s, a, abstract_ns] += p * w
                if not done:
                    # if any state is not terminal we don't consider the abstract state terminal
                    terminal[abstract_ns] = False
    return transition, reward, cost, terminal, abs_map


def abstract(state_id: int, features, feature_domains, env):
    """
    maps an state to its abstract state
    :param state_id:
    :return: abstract_state_id
    """
    encoded_state = list(env.decode(state_id))
    abs_encoded_state = [encoded_state[v] for v in features]
    return encode(abs_encoded_state, feature_domains)


def encode(state_features: list, feature_domains: Sequence[Sequence[int]]):
    i = 0
    for v, value in enumerate(state_features):
        i *= len(feature_domains[v])
        i += value
    return i


def monte_carlo_evaluation(env, agent, horizon, discount_factor=1, number_of_episodes=1000, verbose=False):
    episodes_returns = np.zeros(number_of_episodes)
    episodes_costs = np.zeros(number_of_episodes)
    episodes_length = np.zeros(number_of_episodes)
    episodes_fail = np.zeros(number_of_episodes)
    with trange(number_of_episodes, desc="monte carlo evaluation", unit='episodes', disable=not verbose) as progress:
        for i in progress:
            state = env.reset()
            episode_return = 0
            episode_cost = 0
            steps = 0
            fail = 0
            for t in range(horizon):
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                episode_return += reward * discount_factor ** t
                cost = info.get('cost', 0)
                episode_cost += cost * discount_factor ** t
                # print("episode {:>2} time step {:>2} state {:>2} action {:>2} reward {:>2} cost {:>2} next_state {:>2} done {}".format(i, t, state, action, reward, cost, next_state, done))
                # print('\t', list(env.decode(state)), list(env.decode(next_state)), info)
                state = next_state
                steps += 1
                if done:
                    fail = int('fail' in info and info['fail'])
                    break
            episodes_returns[i] = episode_return
            episodes_costs[i] = episode_cost
            episodes_length[i] = steps
            episodes_fail[i] = fail
            agent.end_episode(evaluation=True)

    return episodes_returns.mean(), episodes_costs.mean(), episodes_length.mean(), episodes_fail.mean()
