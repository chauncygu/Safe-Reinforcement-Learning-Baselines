from gym.envs.toy_text.frozen_lake import FrozenLakeEnv as _FrozenLakeEnv
from gym import spaces
import numpy as np


class FrozenLakeEnv(_FrozenLakeEnv):
    """Modified version of FrozenLake-v0.

    1. Convert integer states to one hot encoding.
    2. Make the goal state reversible
    """
    def __init__(self, map_name):
        super(FrozenLakeEnv, self).__init__(map_name=map_name,
                                            is_slippery=False)
        self.observation_space = spaces.Box(low=np.zeros(self.nS),
                                            high=np.ones(self.nS))
        # Make the goal state not terminate
        goal_s = self.nS - 1
        left_s = goal_s - 1
        up_s = goal_s - int(np.sqrt(self.nS))

        self.P[goal_s] = {
            0: [(1.0, left_s, 0.0, False)],
            1: [(1.0, goal_s, 1.0, True)],
            2: [(1.0, goal_s, 1.0, True)],
            3: [(1.0, up_s, 0.0, True)],
        }

    def _s_to_one_hot(self, s):
        one_hot = np.zeros(self.nS)
        one_hot[s] = 1.
        return one_hot

    def step(self, a):
        (s, r, done, info) = super(FrozenLakeEnv, self).step(a)
        done = (s == self.nS - 1)  # Assume we can't detect dangerous states
        one_hot = self._s_to_one_hot(s)
        r -= 1  # Make the reward be in {-1, 0}
        return (one_hot, r, done, info)

    def reset(self):
        s = super(FrozenLakeEnv, self).reset()
        one_hot = self._s_to_one_hot(s)
        return one_hot
