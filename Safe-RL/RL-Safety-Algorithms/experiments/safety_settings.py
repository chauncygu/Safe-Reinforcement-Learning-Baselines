import os
import argparse


alg_setup = {
    'trpo': {"target_kl": [0.001, 0.01]},
    'lag-trpo': {'target_kl': [1.0e-4, 1.0e-3, 1.0e-2],
                 'lambda_lr': [0.001, 0.01, 0.1]},  # SGD is default
    'cpo': {'target_kl': [1.0e-4, 5.0e-4, 1.0e-3], 'lam_c': [0.50, 0.90, 0.95]},
    'pdo': {'target_kl': [1.0e-4, 1.0e-3, 1.0e-2],
            'lambda_lr': [0.001, 0.01, 0.1]},  # Adam is default
}


def get_alg_setup():
    return alg_setup


def argument_parser():
    n_cpus = os.cpu_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-cores', '-c', type=int, default=n_cpus,
                        help='Number of parallel processes generated.')
    parser.add_argument('--num-runs', '-r', type=int, default=4,
                        help='Number of total runs that are executed.')
    parser.add_argument('--log-dir', type=str, default='/var/tmp/ga87zej',
                        help='Define a custom directory for logging.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Define the initial seed.')
    args = parser.parse_args()
    return args
