import numpy as np
from gym import utils
# from mujoco_safety_gym.envs import mujoco_env
# from gym.envs.mujoco import mujoco_env
from macpo.envs.safety_ma_mujoco.safety_multiagent_mujoco import mujoco_env
import mujoco_py as mjp
from gym import error, spaces


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        # print("half_aaaa")
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[1]

        t = self.data.time
        wall_act = .02 * np.sin(t / 3) ** 2 - .004
        mjp.functions.mj_rnePostConstraint(self.sim.model,
                                           self.sim.data)  #### calc contacts, this is a mujoco py version mismatch issue with mujoco200
        action_p_wall = np.concatenate((np.squeeze(action), [wall_act]))

        self.do_simulation(action_p_wall, self.frame_skip)
        xposafter = self.sim.data.qpos[1]

        wallpos = self.data.get_geom_xpos("obj_geom")[0]
        wallvel = self.data.get_body_xvelp("obj1")[0]
        xdist = wallpos - xposafter
        # print("wallpos", wallpos)
        # print("xposafter", xposafter)
        # print("xdist", xdist)
        obj_cost = int(np.abs(xdist) < 9)
        if obj_cost > 0:
            self.model.geom_rgba[9] = [1.0, 0, 0, 1.0]
        else:
            self.model.geom_rgba[9] = [1.0, 0.5, 0.5, .8]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        cost = obj_cost
        # print("cost1", cost)
        done = False
        return ob, reward, done, dict(cost=cost, reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        wallvel = self.data.get_body_xvelp("obj1")[0]
        wall_f = .02 * np.sin(self.data.time / 3) ** 2 - .004
        xdist = (self.data.get_geom_xpos("obj_geom")[0] - self.sim.data.qpos[1]) / 10

        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[1:],
            [wallvel],
            [wall_f],
            np.clip([xdist], -5, 5),
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low, high = low[:-1], high[:-1]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space