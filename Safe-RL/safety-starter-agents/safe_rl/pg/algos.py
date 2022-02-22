from functools import partial
from safe_rl.pg.agents import PPOAgent, TRPOAgent, CPOAgent
from safe_rl.pg.run_agent import run_polopt_agent

def ppo(**kwargs):
    ppo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = PPOAgent(**ppo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


def ppo_lagrangian(**kwargs):
    # Objective-penalized form of Lagrangian PPO.
    ppo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True
                    )
    agent = PPOAgent(**ppo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


def trpo(**kwargs):
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=False,
                    learn_penalty=False,
                    penalty_param_loss=False  # Irrelevant in unconstrained
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


def trpo_lagrangian(**kwargs):
    # Objective-penalized form of Lagrangian TRPO.
    trpo_kwargs = dict(
                    reward_penalized=False,
                    objective_penalized=True,
                    learn_penalty=True,
                    penalty_param_loss=True
                    )
    agent = TRPOAgent(**trpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)


def cpo(**kwargs):
    cpo_kwargs = dict(
                    reward_penalized=False,  # Irrelevant in CPO
                    objective_penalized=False,  # Irrelevant in CPO
                    learn_penalty=False,  # Irrelevant in CPO
                    penalty_param_loss=False  # Irrelevant in CPO
                    )
    agent = CPOAgent(**cpo_kwargs)
    run_polopt_agent(agent=agent, **kwargs)