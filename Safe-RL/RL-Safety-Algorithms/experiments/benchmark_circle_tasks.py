from rl_safety_algorithms.benchmark import Benchmark
import bullet_safety_gym  # noqa
from safety_settings import alg_setup, argument_parser


def main(args):
    env_specific_kwargs = {
        'SafetyBallCircle-v0': {'epochs': 500, 'steps_per_epoch': 32000},
        'SafetyCarCircle-v0': {'epochs': 500, 'steps_per_epoch': 32000},
        'SafetyDroneCircle-v0': {'epochs': 1000, 'steps_per_epoch': 64000},
        'SafetyAntCircle-v0': {'epochs': 1500, 'steps_per_epoch': 64000},
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
