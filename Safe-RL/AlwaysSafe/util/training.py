import os
from multiprocessing import Pool

import gym

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

from util.mdp import monte_carlo_evaluation


def train_agent(agent, env, number_of_episodes, horizon, seed, out_dir=None, eval_episodes=10, discount_factor=1,
                label='', position=0, verbose=True):
    env.seed(seed)
    env.reset()
    agent.seed(seed)
    np.random.seed(seed)
    results = run_training_episodes(agent, env, number_of_episodes, horizon, eval_episodes, discount_factor,
                                    label, position, verbose=verbose)
    if out_dir is not None:
        save_and_plot(results, out_dir, seed)
    return results


def save_and_plot(data, out_dir, seed, plot=True, window=100, testing=False):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir, 'results_{}.csv'.format(seed)))
    if not plot:
        return
    for key in df.keys():
        if not testing:
            plt.clf()
        plt.plot(df['{}'.format(key)], label="{}".format(key))
        plt.plot(df['{}'.format(key)].rolling(window).mean(), label="{}_rolling_average".format(key))
        plt.savefig(os.path.join(out_dir, '{}_{}.png'.format(key, seed)))


def run_training_episodes(agent, env, number_of_episodes, horizon, eval_episodes=1, discount_factor=1, label='', position=0,
                          log_freq=10, verbose=False, out_dir=None, seed=None):
    training_returns = []
    training_costs = []
    training_length = []
    training_fails = []
    evaluation_returns = []
    evaluation_costs = []
    evaluation_length = []
    evaluation_fail = []
    desc = "training {}".format(label)
    with trange(number_of_episodes, desc=desc, unit="episode", position=position, disable=not verbose) as progress_bar:
        for i in progress_bar:
            state = env.reset()
            t = 0
            episode_return = 0
            episode_cost = 0
            fail = 0
            steps = 0
            for t in range(horizon):
                # print("episode {:>2} time step {:>2} state {:>2}".format(i, t, state))
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                episode_return += reward * discount_factor ** t
                episode_cost += info.get('cost', 0) * discount_factor ** t
                agent.add_transition(state, reward, action, next_state, done, info)
                state = next_state
                steps += 1
                if done:
                    if ('fail' in info) and info['fail']:
                        fail = 1
                    break
            agent.end_episode()

            training_returns.append(episode_return)
            training_costs.append(episode_cost)
            training_length.append(steps)
            training_fails.append(fail)

            if eval_episodes > 0:
                evaluation = monte_carlo_evaluation(env, agent, agent.horizon, discount_factor, eval_episodes)
                evaluation_returns.append(evaluation[0])
                evaluation_costs.append(evaluation[1])
                evaluation_length.append(evaluation[2])
                evaluation_fail.append(evaluation[3])

            if not (i % log_freq):
                progress_bar.set_postfix(
                    t_ret=np.mean(training_returns[-log_freq:]),
                    t_cost=np.mean(training_costs[-log_freq:]),
                    e_ret=np.mean(evaluation_returns[-log_freq:]),
                    e_cost=np.mean(evaluation_costs[-log_freq:]),
                )
                if out_dir is not None:
                    save_and_plot({
                        "training_returns": training_returns,
                        "training_costs": training_costs,
                        "training_length": training_length,
                        "training_fail": training_fails,
                        "evaluation_returns": evaluation_returns,
                        "evaluation_costs": evaluation_costs,
                        "evaluation_length": evaluation_length,
                        "evaluation_fail": evaluation_fail,
                    }, out_dir, seed, plot=False)

    return {
        "training_returns": training_returns,
        "training_costs": training_costs,
        "training_length": training_length,
        "training_fail": training_fails,
        "evaluation_returns": evaluation_returns,
        "evaluation_costs": evaluation_costs,
        "evaluation_length": evaluation_length,
        "evaluation_fail": evaluation_fail,
    }


def run_experiment(env_id, env_kwargs, agent_name, agentClass, agent_kwargs, seed,
                   number_of_episodes, position, out_dir, eval_episodes):
    env = gym.make(env_id, **env_kwargs)
    agent = agentClass.from_discrete_env(env, **agent_kwargs)
    agent_out_dir = os.path.join(out_dir, agent_name)
    train_agent(agent, env, number_of_episodes, agent.horizon, seed, out_dir=agent_out_dir,
                eval_episodes=eval_episodes, label=agent_name, position=position)


def run_experiments_batch(agents, env_id, env_kwargs, eval_episodes, number_of_episodes, out_dir, seeds, parallel=True):
    experiments = []
    for seed in seeds:
        for (agent_name, agentClass, agent_kwargs) in agents:
            position = len(experiments)
            x = (env_id, env_kwargs, agent_name, agentClass, agent_kwargs, seed, number_of_episodes,
                 position, out_dir, eval_episodes)
            experiments.append(x)
            print(x)
    if parallel:
        with Pool() as pool:
            pool.starmap(run_experiment, experiments)
    else:
        for e in experiments:
            run_experiment(*e)
