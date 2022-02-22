#!/usr/bin/env python


# This file is just to get around a baselines import hack.
# env_type is set based on the final part of the entry_point module name.
# In the regular gym mujoco envs this is 'mujoco'.
# We want baselines to treat these as mujoco envs, so we redirect from here,
# and ensure the registry entries are pointing at this file as well.
from safety_gym.envs.engine import *  # noqa
