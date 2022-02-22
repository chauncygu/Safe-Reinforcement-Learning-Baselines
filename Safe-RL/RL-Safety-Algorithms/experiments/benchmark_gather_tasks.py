import os
import argparse
from rl_safety_algorithms.benchmark import Benchmark
import bullet_safety_gym  # noqa
from safety_settings import alg_setup, argument_parser


def main(args):
    env_specific_kwargs = {
        'SafetyBallGather-v0': {'epochs': 500, 'cost_limit': 0.2,
                                'steps_per_epoch': 32000},
        'SafetyCarGather-v0': {'epochs': 500, 'cost_limit': 0.2,
                               'steps_per_epoch': 32000},
        'SafetyDroneGather-v0': {'epochs': 1000, 'cost_limit': 0.2,
                                 'steps_per_epoch': 64000},
        'SafetyAntGather-v0': {'epochs': 1000, 'cost_limit': 0.2,
                               'steps_per_epoch': 64000}
    }
    bench = Benchmark(
        alg_setup,
        env_ids=list(env_specific_kwargs.keys()),
        log_dir=args.log_dir,
        num_cores=args.num_cores,
        num_runs=args.num_runs,
        env_specific_kwargs=env_specific_kwargs,
        use_mpi=True,
        init_seed=args.seed,
    )
    bench.run()


if __name__ == '__main__':
    args = argument_parser()
    main(args)
