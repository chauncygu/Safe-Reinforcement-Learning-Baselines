import numpy as np


class Pendulum:

    def __init__(self):
        self.obs_dim = 3
        self.action_dim = 1
        self.num_vertex = 2
        self.max_speed = 8
        self.max_action = 15.
        self.dt = .05
        self.g = 10.
        self.m = 1.
        self.l = 1.
        self.safe_th = 1.  # safe region [-1, 1]

        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_action, self.max_action)[0]

        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([1, 0])  # initialize in the safe region
        # high = np.array([np.pi, 1])         # initialize
        self.state = np.random.uniform(low=-high, high=high)

        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def get_action_vertex(self, obs):
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        th = np.arccos(obs[:, 0]) * np.sign(obs[:, 1])
        thdot = obs[:, 2]
        action_u = (+self.safe_th - th - thdot * dt - 3*g/(2*l) * np.sin(th) * (dt**2)) / (3*(dt**2) / (m * (l**2)))
        action_l = (-self.safe_th - th - thdot * dt - 3*g/(2*l) * np.sin(th) * (dt**2)) / (3*(dt**2) / (m * (l**2)))

        action_u = np.clip(action_u, -self.max_action, self.max_action)
        action_l = np.clip(action_l, -self.max_action, self.max_action)

        return np.expand_dims(np.stack(zip(action_l, action_u)), axis=2)


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
