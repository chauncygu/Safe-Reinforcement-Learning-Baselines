"""Wrapper for OpenAI Gym."""

from SafeRLBench import EnvironmentBase
from SafeRLBench.error import add_dependency

try:
    import gym
except ModuleNotFoundError:
    gym = None


# TODO: GymWrap: Add examples to docs
class GymWrap(EnvironmentBase):
    """Wrapper class for the OpenAI Gym.

    Attributes
    ----------
    env : gym environment
        Environment of the OpenAI Gym created by gym.make().
    horizon : integer
        Horizon for rollout.
    render : boolean
        Default: False. If True simulation will be rendered during rollouts on
        this instance.

    Notes
    -----
    The GymWrap class relies on the complete observability of the state
    through a state field in the respective gym environment. For the classic
    control problem this is indeed the case, but on other environment it
    remains to be untested.
    """

    def __init__(self, env, horizon=100, render=False):
        """Initialize attributes.

        Parameters
        ----------
        env : gym environment
            Instance of the gym environment that should be optimized on.
        horizon : integer
            Horizon for rollout.
        render : boolean
            Default: False ; If True simulation will be rendered during
            rollouts on this instance.
        """
        add_dependency(gym, 'Gym')

        EnvironmentBase.__init__(self, env.observation_space, env.action_space,
                                 horizon)
        self.environment = env.unwrapped
        self.render = render
        self.done = False

        self.environment.reset()

    def _update(self, action):
        observation, reward, done, info = self.environment.step(action)
        self.done = done
        return action, observation, reward

    def _reset(self):
        self.environment.reset()
        self.done = False

    def _rollout(self, policy):
        trace = []
        for n in range(self.horizon):
            if self.render:
                self.environment.render()
            trace.append(self.update(policy(self.state)))
            if self.done:
                break
        return trace

    @property
    def state(self):
        """Observable system state."""
        return self.environment.state

    @state.setter
    def state(self, s):
        assert self.state_space.contains(s)
        self.environment.state = s


def _get_test_args():
    return [gym.make('MountainCar-v0')]
