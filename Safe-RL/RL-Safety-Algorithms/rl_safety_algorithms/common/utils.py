import argparse
import datetime
import os
import sys

import gym
import warnings
import numpy as np
from collections import defaultdict
import re
from importlib import import_module


def get_alg_module(alg, *submodules):
    """ inspired by source: OpenAI's baselines."""

    if submodules:
        mods = '.'.join(['rl_safety_algorithms', 'algs', alg, *submodules])
        alg_module = import_module(mods)
    else:
        alg_module = import_module('.'.join(['rl_safety_algorithms', 'algs', alg, alg]))

    return alg_module


def get_alg_class(alg, env_id, **kwargs):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    alg_cls_init = getattr(alg_mod, 'get_alg')

    return alg_cls_init(env_id, **kwargs)


def get_learn_function(alg):
    """Get the learn function of a particular algorithm."""
    alg_mod = get_alg_module(alg)
    learn_func = getattr(alg_mod, 'learn')

    return learn_func


def get_env_type(env_id: str):
    """Determines the type of the environment if there is no args.env_type.

    source: OpenAI's Baselines Repository

    Parameters
    ----------
    env_id:
        Name of the gym environment.

    Returns
    -------
    env_type: str
    env_id: str
    """
    all_registered_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        try:
            env_type = env.entry_point.split(':')[0].split('.')[-1]
            all_registered_envs[env_type].add(env.id)
        except AttributeError:
            print('Could not fetch ', env.id)

    # Re-parse the gym registry, since we could have new data
    # since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        all_registered_envs[env_type].add(env.id)

    if env_id in all_registered_envs.keys():
        env_type = env_id
        env_id = [g for g in all_registered_envs[env_type]][0]
    else:
        env_type = None
        for g, e in all_registered_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(
            env_id,
            all_registered_envs.keys())

    return env_type, env_id


def get_defaults_kwargs(alg, env_id):
    """ inspired by OpenAI's baselines."""
    env_type, _ = get_env_type(env_id=env_id)

    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        pass
        # warnings.warn(
        #     f'Could not fetch default kwargs for env_type: {env_type}')
        # Fetch standard arguments from locomotion environments
        try:  # fetch from defaults()
            env_type = 'defaults'
            alg_defaults = get_alg_module(alg, 'defaults')
            kwargs = getattr(alg_defaults, env_type)()
        except:
            env_type = 'locomotion'
            alg_defaults = get_alg_module(alg, 'defaults')
            kwargs = getattr(alg_defaults, env_type)()

    return kwargs


def convert_to_string_only_dict(input_dict):
    """
    Convert all values of a dictionary to string objects
    Useful, if you want to save a dictionary as .json file to the disk

    :param input_dict: dict, input to be converted
    :return: dict, converted string dictionary
    """
    converted_dict = dict()
    for key, value in input_dict.items():
        if isinstance(value, dict):  # transform dictionaries recursively
            converted_dict[key] = convert_to_string_only_dict(value)
        elif isinstance(value, type):
            converted_dict[key] = str(value.__name__)
        else:
            converted_dict[key] = str(value)
    return converted_dict


def get_default_args(debug_level=0,
                     env='CartPole-v0',
                     func_name='testing',
                     log_dir='/var/tmp/ga87zej/',
                     threads=os.cpu_count()
                     ):
    """ create the default arguments for program execution
    :param threads: int, number of available threads
    :param env: str, name of RL environment
    :param func_name:
    :param log_dir: str, path to directory where logging files are going to be created
    :param debug_level: 
    :return: 
    """
    if sys.version_info[0] < 3:
        raise Exception("Must be using Python 3")

    parser = argparse.ArgumentParser(description='This is the default parser.')
    parser.add_argument('--alg', default=os.cpu_count(), type=int,
                        help='Algorithm to use (in case of a RL problem. (default: PPO)')
    parser.add_argument('--threads', default=threads, type=int,
                        help='Number of available Threads on CPU.')
    parser.add_argument('--debug', default=debug_level, type=int,
                        help='Debug level (0=None, 1=Low debug prints 2=all debug prints).')
    parser.add_argument('--env', default=env, type=str,
                        help='Default environment for RL algorithms')
    parser.add_argument('--func', dest='func', default=func_name,
                        help='Specify function name to be testing')
    parser.add_argument('--log', dest='log_dir', default=log_dir,
                        help='Set the seed for random generator')

    args = parser.parse_args()
    args.log_dir = os.path.abspath(os.path.join(args.log_dir,
                                                datetime.datetime.now().strftime(
                                                    "%Y_%m_%d__%H_%M_%S")))
    return args


def get_seed_from_sys_args():
    _seed = 0
    for pos, elem in enumerate(
            sys.argv):  # look if there exists seed=X in _args,
        if len(elem) > 5 and elem[:5] == 'seed=':
            _seed = int(elem[5:])
    return _seed


def normalize(xs,
              axis=None,
              eps=1e-8):
    """ Normalize array along axis
    :param xs: np.array(), array to normalize
    :param axis: int, axis along which is normalized
    :param eps: float, offset to avoid division by zero
    :return: np.array(), normed array
    """
    return (xs - xs.mean(axis=axis)) / (xs.std(axis=axis) + eps)


def make_env(env_name,
             seed=None):
    """ Creates a Gym Environment
    :param env_name: str, name of environment
    :param seed: int, make experiments deterministic
    :return:
    """
    import gym
    from gym import envs
    try:  # try to import Martin's environments
        from mygymenvs.environmentwrapper import get_data_wrapper
    except ImportError:
        print(
            'WARNING: could not import mygymenvs! Hint: Did you install the progressbar package?')
    try:  # try to import roboschool environments
        import roboschool
    except ImportError:
        print('WARNING: could not import roboschool!')
    try:
        import pybullet_envs
        import pybulletgym.envs
    except ImportError:
        print('WARNING: could not import pybullet_envs!')

    if seed:
        tf.random.set_seed(seed)
        np.random.seed(seed=seed)
    all_gym_environments = [env_spec.id for env_spec in envs.registry.all()]

    if env_name in all_gym_environments:
        return gym.make(env_name)
    else:
        raise ValueError(
            'Did not find environment with name: {}'.format(env_name))


def mkdir(path):
    """ create directory at a given path
    :param path: str, path
    :return: bool, True if created directories
    """
    created_dir = False
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        created_dir = True
    return created_dir


def parse_to_sacred_experiment_args(arguments: list, _seed: int,
                                    log_dir: str) -> list:
    """Add seed to sys args if not already defined.

    This brings arguments into the expected structure of the sacred framework.
    Remind: arguments is a list, so this is a call by reference

    Parameters
    ----------
    arguments
        An object holding information of the parsed console arguments. This is a call by reference!
    _seed
        The seed for the random generator.
    log_dir
        Over-write the default BASEDIR by setting a sacred flag.

    Returns
    -------
    list
        The parsed arguments with the expected structure.
    """
    seed_in_args = None
    with_in_args = False
    custom_log_dir = None if log_dir == '' else log_dir

    _args = arguments.copy()  # copy argument list, since this is a call by reference

    # sacred expects the first element of list to be a command
    if len(_args) > 0 and _args[0] == 'with':
        _args = ['run', ] + _args
    if len(_args) == 0:
        _args.append('run')

    for pos, elem in enumerate(_args):  # look if there exists seed=X in _args,
        if len(elem) > 5 and elem[:5] == 'seed=':
            seed_in_args = pos
            value = elem[5:]
            print(
                'WARNING: Seed==={} was found in args, change seed to: {}'.format(
                    value, _seed))
        if len(elem) == 4 and elem == 'with':
            with_in_args = True

    string_argument = 'seed=' + str(_seed)
    if seed_in_args:
        _args[seed_in_args] = string_argument

    else:
        if not with_in_args:
            _args.append('with')
        _args.append(string_argument)

    # over-write the default BASEDIR with custom log dir if provided
    if custom_log_dir:
        _args.append(f'base_dir={custom_log_dir}')

    return _args


def safe_mean(xs):
    """ Calculate mean value of an array safely and avoid division errors
    :param xs: np.array, array to calculate mean
    :return: np.float, mean value of xs
    """
    return np.nan if len(xs) == 0 else float(np.mean(xs))


def safe_std(xs):
    return np.std(xs) if safe_mean(xs) is not np.nan else np.nan
