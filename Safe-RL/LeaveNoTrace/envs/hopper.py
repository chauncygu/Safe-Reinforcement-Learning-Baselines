from gym.envs.mujoco.hopper import HopperEnv as _HopperEnv


class HopperEnv(_HopperEnv):
    """Modified version of Hopper-v1."""

    def step(self, action):
        (obs, r, done, info) = super(HopperEnv, self).step(action)
        return (obs, r, False, info)
