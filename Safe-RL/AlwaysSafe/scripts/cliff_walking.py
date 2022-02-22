import os

from agents import OptCMDPAgent
from agents import AbsOptCMDPAgent
from util.training import run_experiments_batch


def main():
    c = 2
    h = 15
    number_of_episodes = 5000
    seeds = range(100)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    eval_episodes = 1000
    env_id = "gym_factored:cliff_walking_cost-v0"
    env_kwargs = {"num_rows": 4, "num_cols": 6}

    agents = [
        ('OptCMDP', OptCMDPAgent, {
            'cost_bound': c,
            'horizon': h
        }),
        ('AbsOptCMDP ground', AbsOptCMDPAgent, {
            'features': [0, 1],
            'cost_bound': c,
            'horizon': h,
            'policy_type': 'ground'
        }),
        # ('AbsOptCMDP abs', AbsOptCMDP, {
        #     'features': [0, 1],
        #     'cost_bound': c,
        #     'horizon': h,
        #     'policy_type': 'abs'
        # }),
        # ('SafeAbsOptCMDP (global)', AbsOptCMDP, {
        #     'features': [0, 1],
        #     'cost_bound': c,
        #     'horizon': h,
        #     'policy_type': 'global_test'
        # }),
        # ('SafeAbsOptCMDP (global) .9ĉ', AbsOptCMDP, {
        #     'features': [0, 1],
        #     'cost_bound': c,
        #     'horizon': h,
        #     'policy_type': 'global_test',
        #     'cost_bound_coefficient': 0.9
        # }),
        # ('SafeAbsOptCMDP (adaptive)', AbsOptCMDP, {
        #     'features': [0, 1],
        #     'cost_bound': c,
        #     'horizon': h,
        #     'policy_type': 'adaptive'
        # }),
    ]
    run_experiments_batch(agents, env_id, env_kwargs, eval_episodes, number_of_episodes, out_dir, seeds)


if __name__ == '__main__':
    main()
