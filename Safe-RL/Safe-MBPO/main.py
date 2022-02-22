from pathlib import Path

import numpy as np
np.set_printoptions(precision=3, linewidth=120)

from src import cli
from src.defaults import ROOT_DIR
from src.log import default_log as log
from src.checkpoint import CheckpointableData, Checkpointer
from src.config import BaseConfig, Require
from src.torch_util import device
from src.shared import get_env
from src.smbpo import SMBPO


ROOT_DIR = Path(ROOT_DIR)
SAVE_PERIOD = 5


class Config(BaseConfig):
    env_name = Require(str)
    seed = 1
    epochs = 1000
    alg_cfg = SMBPO.Config()


def main(cfg):
    env_factory = lambda: get_env(cfg.env_name)
    data = CheckpointableData()
    alg = SMBPO(cfg.alg_cfg, env_factory, data)
    alg.to(device)
    checkpointer = Checkpointer(alg, log.dir, 'ckpt_{}.pt')
    data_checkpointer = Checkpointer(data, log.dir, 'data.pt')

    # Check if existing run
    if data_checkpointer.try_load():
        log('Data load succeeded')
        loaded_epoch = checkpointer.load_latest(list(range(0, cfg.epochs, SAVE_PERIOD)))
        if isinstance(loaded_epoch, int):
            assert loaded_epoch == alg.epochs_completed
            log('Solver load succeeded')
        else:
            assert alg.epochs_completed == 0
            log('Solver load failed')
    else:
        log('Data load failed')

    if alg.epochs_completed == 0:
        alg.setup()

        # So that we can compare to the performance of randomly initialized policy
        alg.evaluate()

    while alg.epochs_completed < cfg.epochs:
        log(f'Beginning epoch {alg.epochs_completed+1}')
        alg.epoch()
        alg.evaluate()

        if alg.epochs_completed % SAVE_PERIOD == 0:
            checkpointer.save(alg.epochs_completed)
            data_checkpointer.save()


if __name__ == '__main__':
    cli.main(Config(), main)