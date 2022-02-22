import os

from agents import OptCMDPAgent
from agents import AbsOptCMDPAgent
from util.training import run_experiments_batch


def main():
    c = 3
    h = 6
    number_of_episodes = 100
    seeds = range(100)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    eval_episodes = 1000
    env_id = 'gym_factored:cmdp-v0'
    env_kwargs = {}

    agents = [
        ('OptCMDP', OptCMDPAgent, {
            'cost_bound': c,
            'horizon': h
        }),
        ('AbsOptCMDP ground', AbsOptCMDPAgent, {
            'features': [],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground'
        }),
        ('AbsOptCMDP abs', AbsOptCMDPAgent, {
            'features': [],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'abs'
        }),
        ('SafeAbsOptCMDP (global)', AbsOptCMDPAgent, {
            'features': [],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'global_test'
        }),
        ('SafeAbsOptCMDP (global) .9ĉ', AbsOptCMDPAgent, {
            'features': [],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'global_test',
            'cost_bound_coefficient': 0.9
        }),
        ('SafeAbsOptCMDP (adaptive)', AbsOptCMDPAgent, {
            'features': [],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'adaptive'
        }),
    ]
    run_experiments_batch(agents, env_id, env_kwargs, eval_episodes, number_of_episodes, out_dir, seeds)


if __name__ == '__main__':
    main()
