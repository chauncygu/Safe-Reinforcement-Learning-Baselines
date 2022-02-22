from __future__ import division, print_function, absolute_import
import argparse


def safemdp_argparse():

    parser = argparse.ArgumentParser(description='Safe-NearOpt-MDP')
    parser.add_argument('--h', type=float, default=-0.25,
                        help='safety threshold')
    parser.add_argument('--L', type=float, default=0.0,
                        help='Lipschitz constant')
    parser.add_argument('--noise-reward', type=float, default=0.001,
                        help='noise for safety')
    parser.add_argument('--noise-safety', type=float, default=0.001,
                        help='noise for reward')
    parser.add_argument('--beta-safety', type=float, default=2.0,
                        help='Scaling factor for confidence interval (safety)')
    parser.add_argument('--beta-reward', type=float, default=3.0,
                        help='Scaling factor for confidence interval (reward)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--world-shape', type=float, default=(20, 20),
                        help='world size')
    parser.add_argument('--step-size', type=float, default=(0.5, 0.5),
                        help='step size')
    parser.add_argument('--n-samples', type=int, default=1,
                        help='number of samples used for initializing GP')
    parser.add_argument('--freq-check', type=int, default=1,
                        help='frequeny of check if safe-exploration can finish')
    parser.add_argument('--mdp-alg', type=str, default='pi',
                        help='algorithm to solve an MDP')
    parser.add_argument('--max-iter-safe-exp', type=int, default=100,
                        help='max num of iterations for exploration of safety')
    parser.add_argument('--max-iter-reward-opt', type=int, default=300,
                        help='max number of iterations for optimizing reward')
    parser.add_argument('--max-time-steps', type=int, default=2000,
                        help='max number of time steps')
    parser.add_argument('--thres-ci', type=float, default=0.005,
                        help='thresh. for stopping exploration w.r.t. CI')
    parser.add_argument('--multi-obs', default=True,
                        help='define if the agent can observe multiple points')
    parser.add_argument('--render-gym', default=True,
                        help='rendering using GP-Safety-Gym')
    parser.add_argument('--es2-type', type=str, default='es2',
                        choices=['es2', 'p_es2', 'none'],
                        help='whether or not ES2/P-ES2 is used')
    args = parser.parse_args()

    return args
