from rl_safety_algorithms.common.model import Model
import argparse
import os
import getpass
import psutil
import sys
import time
import warnings
import rl_safety_algorithms.common.mpi_tools as mpi_tools

try:
    import pybullet_envs  # noqa
except ImportError:
    warnings.warn('pybullet_envs package not found.')
try:
    import bullet_safety_gym  # noqa
except ImportError:
    warnings.warn('Bullet-Safety-Gym package not found.')


if __name__ == '__main__':
    physical_cores = psutil.cpu_count(logical=False)  # exclude hyper-threading
    # Seed must be < 2**32 => use 2**16 to allow seed += 10000*proc_id() for MPI
    random_seed = int(time.time()) % 2**16
    user_name = getpass.getuser()
    default_log_dir = os.path.join('/var/tmp/', user_name)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--alg', type=str, required=True,
                        help='Choose from: {iwpg, ppo, trpo, npg}')
    parser.add_argument('--cores', '-c', type=int, default=physical_cores,
                        help=f'Number of cores used for calculations.')
    parser.add_argument('--debug', action='store_true',
                        help='Show debug prints during training.')
    parser.add_argument('--env', type=str, required=True,
                        help='Example: HopperBulletEnv-v0')
    parser.add_argument('--no-mpi', action='store_true',
                        help='Do not use MPI for parallel execution.')
    parser.add_argument('--play', action='store_true',
                        help='Visualize agent after training.')
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Number of total runs that are executed.')
    parser.add_argument('--seed', default=random_seed, type=int,
                        help=f'Define the init seed, e.g. {random_seed}')
    parser.add_argument('--search', action='store_true',
                        help='If given search over learning rates.')
    parser.add_argument('--log-dir', type=str, default=default_log_dir,
                        help='Define a custom directory for logging.')

    args, unparsed_args = parser.parse_known_args()
    # Use number of physical cores as default. If also hardware threading CPUs
    # should be used, enable this by the use_number_of_threads=True
    use_number_of_threads = True if args.cores > physical_cores else False
    if mpi_tools.mpi_fork(args.cores,
                          use_number_of_threads=use_number_of_threads):
        # Re-launches the current script with workers linked by MPI
        sys.exit()
    print('Unknowns:', unparsed_args) if mpi_tools.proc_id() == 0 else None

    model = Model(
        alg=args.alg,
        env_id=args.env,
        log_dir=args.log_dir,
        init_seed=args.seed,
        unparsed_args=unparsed_args,
        use_mpi=not args.no_mpi
    )
    model.compile(num_runs=args.runs, num_cores=args.cores)
    model.fit()
    model.eval()
    if args.play:
        model.play()
