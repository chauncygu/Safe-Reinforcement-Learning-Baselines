import numpy as np


class Hovercraft:

    def __init__(self, dummy_vertex=False, fixed_init=False, fixed_target=False, safe_th=0.25, lamb=0):
        self.obs_dim = 8
        self.action_dim = 2
        self.num_vertex = 5
        self.max_action = 20.
        self.max_speed = 10.
        self.max_th = np.pi / 2
        self.max_thdot = np.pi
        self.dt = .05
        self.g = 10.
        self.safe_xu = 10. 
        self.safe_xl = 0.
        self.safe_yu = 10.  
        self.safe_yl = 0.
        self.safe_th = safe_th  # safe theta region [-0.25, 0.25]
        self.t = 0  # clock time
        self.dummy_vertex = dummy_vertex
        self.fixed_init = fixed_init
        self.fixed_target = fixed_target
        self.init_state = np.array([0, 0, 0, 0, 0, 0])  # default initial state
        self.state = self.init_state
        self.target = np.array([5, 5])  # default target
        self.lamb = lamb  # reward trade off

        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, u):
        x, y, th, xdot, ydot, thdot = self.state
        x_target, y_target = self.target
        u = np.clip(u, 0, self.max_action)

        costs = (x - x_target)**2 + (y - y_target)**2 + self.lamb * th**2 + .1 * (xdot**2 + ydot**2 + self.lamb * thdot**2) + .001 * (u[0]**2 + u[1]**2)

        new_xdot = xdot + (u[0] + u[1]) * np.sin(th) * self.dt
        new_ydot = ydot + ((u[0] + u[1]) * np.cos(th) - self.g) * self.dt
        new_thdot = thdot + (u[0] - u[1]) * self.dt

        new_x = x + .5 * new_xdot * self.dt
        new_y = y + .5 * new_ydot * self.dt
        new_th = th + .5 * new_thdot * self.dt

        new_th = np.clip(new_th, -self.max_th, self.max_th)
        new_xdot = np.clip(new_xdot, -self.max_speed, self.max_speed)
        new_ydot = np.clip(new_ydot, -self.max_speed, self.max_speed)
        new_thdot = np.clip(new_thdot, -self.max_thdot, self.max_thdot)

        self.state = np.array([new_x, new_y, new_th, new_xdot, new_ydot, new_thdot])

        self.t += self.dt
        if not self.fixed_target and self.t % 5 == 0:
            self.set_random_target()  # reset target every 5 seconds

        return self._get_obs(), -costs, False, {}

    def reset(self):
        self.t = 0
        if self.fixed_init:
            self.state = self.init_state
        else:
            high = np.array([self.safe_xu, self.safe_yu, self.safe_th, 0, 0, 0])  # initialization range
            low = np.array([self.safe_xl, self.safe_yl, -self.safe_th, 0, 0, 0])
            self.state = np.random.uniform(low=low, high=high)
        if not self.fixed_target:
            self.set_random_target()

        return self._get_obs()

    def set_random_target(self):
        high = np.array([self.safe_xu, self.safe_yu])
        low = np.array([self.safe_xl, self.safe_yl])
        self.target = np.random.uniform(low=low, high=high)

    def _get_obs(self):
        return np.concatenate([self.state, self.target])  # [x, y, th, xdot, ydot, thdot, target_x, target_y]

    def get_action_vertex(self, obs):
        batch_size = obs.shape[0]

        # triangle shape vertices
        v0 = np.array([0, 0])  # origin
        v1 = v2 = np.array([0, self.max_action])   # up angle
        v3 = v4 = np.array([self.max_action, 0])   # right angle
        vertices = np.tile(np.stack([v0, v1, v2, v3, v4]), (batch_size, 1, 1))
        if self.dummy_vertex:
            return vertices

        th = obs[:, 2]
        thdot = obs[:, 5]

        udiff_u = - 2 * (-self.safe_th - th - self.dt * thdot) / (self.dt**2)
        udiff_l = - 2 * (+self.safe_th - th - self.dt * thdot) / (self.dt**2)

        udiff_u = np.clip(udiff_u, -self.max_action, self.max_action)
        udiff_l = np.clip(udiff_l, -self.max_action, self.max_action)

        vertices[:, 0, 0] = np.maximum(0, -udiff_u)  # vertex 0 x
        vertices[:, 0, 1] = np.maximum(0, udiff_l)  # vertex 0 y
        vertices[:, 1, 0] = np.maximum(0, -udiff_u)  # vertex 1 x
        vertices[:, 1, 1] = np.maximum(udiff_u, 0)  # vertex 1 y
        vertices[:, 2, 0] = .5 * (self.max_action - udiff_u)  # vertex 2 x
        vertices[:, 2, 1] = .5 * (self.max_action + udiff_u)  # vertex 2 y
        vertices[:, 3, 0] = .5 * (self.max_action - udiff_l)  # vertex 3 x
        vertices[:, 3, 1] = .5 * (self.max_action + udiff_l)  # vertex 3 y
        vertices[:, 4, 0] = np.maximum(-udiff_l, 0)  # vertex 4 x
        vertices[:, 4, 1] = np.maximum(0, udiff_l)  # vertex 4 y

        return vertices

