from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import random

import torch

from .defaults import ROOT_DIR
from .config import TaggedUnion
from .log import default_log as log
from .util import set_seed, random_string


ROOT_DIR = Path(ROOT_DIR)
assert ROOT_DIR.is_dir(), ROOT_DIR


def yes_no_prompt(prompt, default=None):
    yes_str, no_str = 'y', 'n'
    if default == 'y':
        yes_str = '[y]'
    elif default == 'n':
        no_str = '[n]'
    elif default is None:
        pass
    else:
        raise ValueError('default argument to yes_no_prompt must be \'y\', \'n\', or None')

    response = input(f'{prompt} {yes_str}/{no_str} ')
    if response == 'y':
        return True
    elif response == 'n':
        return False
    elif response == '':
        if default == 'y':
            return True
        elif default == 'n':
            return False
        raise RuntimeError('No response to yes/no prompt, and no default given')
    else:
        raise RuntimeError('Invalid response to yes/no prompt (expected y, n, or nothing)')


def try_parse(s):
    try:
        return eval(s)
    except:
        return s


def main(cfg, main):
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default=[], action='append')
    parser.add_argument('-s', '--set', default=[], action='append', nargs=2)
    parser.add_argument('--run-dir', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    cli_args = parser.parse_args()
    set_args = dict(cli_args.set)

    # Directory structure: ROOT_DIR / logs / env_name / run-dir
    root_log_dir = ROOT_DIR / ('debug_logs' if cli_args.debug else 'logs')

    if cli_args.resume:
        assert 'env_name' in set_args, 'Must specify env_name if using --resume'
        assert cli_args.run_dir is not None, 'Must specify --run-dir if using --resume'
        run_dir = root_log_dir / set_args['env_name'] / cli_args.run_dir
        assert run_dir.is_dir(), f'Run directory does not exist: {run_dir}'

        with (run_dir / 'config.json').open('r') as f:
            saved_cfg = json.load(f)
            cfg.update(saved_cfg)
    else:
        for cfg_path in cli_args.config:
            with Path(cfg_path).open('r') as f:
                cfg.update(json.load(f))

        for dot_path, value in set_args.items():
            cfg.nested_set(dot_path.split('.'), try_parse(value))

    # Ensure all required arguments have been set
    cfg.verify()
    for attr in ('env_name', 'seed'):
        assert hasattr(cfg, attr), f'Config must specify {attr}'

    env_dir = root_log_dir / cfg.env_name
    env_dir.mkdir(exist_ok=True, parents=True)

    if cli_args.run_dir is None:
        now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
        random.seed()
        rand_str = random_string(4, include_uppercase=False, include_digits=False)
        run_dir = f'{now_str}_{rand_str}'
    else:
        run_dir = cli_args.run_dir

    log_dir = env_dir / run_dir
    log_dir.mkdir(exist_ok=True)
    log.setup(log_dir)
    log.message(f'Log directory: {log_dir}')

    # Dump config to file
    with (log_dir / 'config.json').open('w') as f:
        json.dump(cfg.vars_recursive(), f, indent=2)

    set_seed(cfg.seed)
    torch.set_num_threads(1)
    main(cfg)